module fortchuck_linalg
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
  use fortchuck_kinds, only: dp, i32, i64
  implicit none (type, external)
  private

  public :: fc_gemm_nn
  public :: fc_gemm_tn
  public :: fc_gemm_nt
  public :: fc_syrk_tn
  public :: fc_frob_inner
  public :: fc_frob_norm
  public :: fc_lu_factor
  public :: fc_lu_solve
  public :: fc_lu_solve_trans
  public :: fc_lu_solve_right
  public :: fc_geqrf_pos

#ifdef FORTCHUCK_USE_BLAS
  interface
    pure subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      import :: dp, i32
      implicit none (type, external)
      character(len=1), intent(in) :: transa, transb
      integer(i32), intent(in)     :: m, n, k, lda, ldb, ldc
      real(dp), intent(in)         :: alpha, beta
      real(dp), intent(in)         :: a(lda, *), b(ldb, *)
      real(dp), intent(inout)      :: c(ldc, *)
    end subroutine dgemm
  end interface
#endif

contains

  !---------------------------------------------------------------------------
  ! C(1:m, 1:n) := alpha * A(1:m, 1:k) * B(1:k, 1:n) + beta * C(1:m, 1:n)
  ! L1/L2 cache-tiled (MC=128, NC=64, KC=128) with 1x4 column-unrolled SIMD
  ! FMA register micro-kernel and unit-stride i inner loop (0 stack arrays).
  !---------------------------------------------------------------------------
  pure subroutine fc_gemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    integer(i32), intent(in) :: m, n, k, lda, ldb, ldc
    real(dp), intent(in)     :: alpha, beta
    real(dp), intent(in)     :: A(lda, *), B(ldb, *)
    real(dp), intent(inout)  :: C(ldc, *)

    integer(i32), parameter :: MC = 128_i32, NC = 64_i32, KC = 128_i32
    integer(i32) :: i, j, l, i0, j0, l0, i_end, j_end, l_end, j_lim4
    real(dp)     :: b0, b1, b2, b3, a_il

    if (m <= 0_i32 .or. n <= 0_i32) return

#ifdef FORTCHUCK_USE_BLAS
    if (int(m, i64) * int(n, i64) * int(max(k, 0_i32), i64) > 8192_i64) then
      call dgemm('N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
      return
    end if
#endif

    if (.not. (abs(beta) > 0.0_dp)) then
      do j = 1_i32, n
        do i = 1_i32, m
          C(i, j) = 0.0_dp
        end do
      end do
    else if (beta < 1.0_dp .or. beta > 1.0_dp) then
      do j = 1_i32, n
        do i = 1_i32, m
          C(i, j) = beta * C(i, j)
        end do
      end do
    end if

    if (k <= 0_i32 .or. .not. (abs(alpha) > 0.0_dp)) return

    do j0 = 1_i32, n, NC
      j_end  = min(j0 + NC - 1_i32, n)
      j_lim4 = j0 + ((j_end - j0 + 1_i32) / 4_i32) * 4_i32 - 1_i32
      do l0 = 1_i32, k, KC
        l_end = min(l0 + KC - 1_i32, k)
        do i0 = 1_i32, m, MC
          i_end = min(i0 + MC - 1_i32, m)

          do j = j0, j_lim4, 4_i32
            do l = l0, l_end
              b0 = alpha * B(l, j)
              b1 = alpha * B(l, j + 1_i32)
              b2 = alpha * B(l, j + 2_i32)
              b3 = alpha * B(l, j + 3_i32)
              do i = i0, i_end
                a_il            = A(i, l)
                C(i, j)         = C(i, j)         + a_il * b0
                C(i, j + 1_i32) = C(i, j + 1_i32) + a_il * b1
                C(i, j + 2_i32) = C(i, j + 2_i32) + a_il * b2
                C(i, j + 3_i32) = C(i, j + 3_i32) + a_il * b3
              end do
            end do
          end do

          do j = j_lim4 + 1_i32, j_end
            do l = l0, l_end
              b0 = alpha * B(l, j)
              if (abs(b0) > 0.0_dp) then
                do i = i0, i_end
                  C(i, j) = C(i, j) + A(i, l) * b0
                end do
              end if
            end do
          end do
        end do
      end do
    end do
  end subroutine fc_gemm_nn

  !---------------------------------------------------------------------------
  ! C(1:m, 1:n) := alpha * A(1:k, 1:m)^T * B(1:k, 1:n) + beta * C(1:m, 1:n)
  ! L1/L2 cache-tiled (MC=64, NC=64, KC=128) with explicit 4x4 scalar-register
  ! FMA micro-kernel over unit-stride l (0 stack arrays).
  !---------------------------------------------------------------------------
  pure subroutine fc_gemm_tn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    integer(i32), intent(in) :: m, n, k, lda, ldb, ldc
    real(dp), intent(in)     :: alpha, beta
    real(dp), intent(in)     :: A(lda, *), B(ldb, *)
    real(dp), intent(inout)  :: C(ldc, *)

    integer(i32), parameter :: MC = 64_i32, NC = 64_i32, KC = 128_i32
    integer(i32) :: i, j, l, i0, j0, l0, i_end, j_end, l_end, i_lim4, j_lim4
    real(dp)     :: a0, a1, a2, a3, b0, b1, b2, b3
    real(dp)     :: c00, c10, c20, c30, c01, c11, c21, c31
    real(dp)     :: c02, c12, c22, c32, c03, c13, c23, c33

    if (m <= 0_i32 .or. n <= 0_i32) return

#ifdef FORTCHUCK_USE_BLAS
    if (int(m, i64) * int(n, i64) * int(max(k, 0_i32), i64) > 8192_i64) then
      call dgemm('T', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
      return
    end if
#endif

    if (.not. (abs(beta) > 0.0_dp)) then
      do j = 1_i32, n
        do i = 1_i32, m
          C(i, j) = 0.0_dp
        end do
      end do
    else if (beta < 1.0_dp .or. beta > 1.0_dp) then
      do j = 1_i32, n
        do i = 1_i32, m
          C(i, j) = beta * C(i, j)
        end do
      end do
    end if

    if (k <= 0_i32 .or. .not. (abs(alpha) > 0.0_dp)) return

    do j0 = 1_i32, n, NC
      j_end  = min(j0 + NC - 1_i32, n)
      j_lim4 = j0 + ((j_end - j0 + 1_i32) / 4_i32) * 4_i32 - 1_i32
      do l0 = 1_i32, k, KC
        l_end = min(l0 + KC - 1_i32, k)
        do i0 = 1_i32, m, MC
          i_end  = min(i0 + MC - 1_i32, m)
          i_lim4 = i0 + ((i_end - i0 + 1_i32) / 4_i32) * 4_i32 - 1_i32

          do j = j0, j_lim4, 4_i32
            do i = i0, i_lim4, 4_i32
              c00 = 0.0_dp; c10 = 0.0_dp; c20 = 0.0_dp; c30 = 0.0_dp
              c01 = 0.0_dp; c11 = 0.0_dp; c21 = 0.0_dp; c31 = 0.0_dp
              c02 = 0.0_dp; c12 = 0.0_dp; c22 = 0.0_dp; c32 = 0.0_dp
              c03 = 0.0_dp; c13 = 0.0_dp; c23 = 0.0_dp; c33 = 0.0_dp
              do l = l0, l_end
                a0  = A(l, i)
                a1  = A(l, i + 1_i32)
                a2  = A(l, i + 2_i32)
                a3  = A(l, i + 3_i32)
                b0  = B(l, j)
                b1  = B(l, j + 1_i32)
                b2  = B(l, j + 2_i32)
                b3  = B(l, j + 3_i32)
                c00 = c00 + a0 * b0; c10 = c10 + a1 * b0; c20 = c20 + a2 * b0; c30 = c30 + a3 * b0
                c01 = c01 + a0 * b1; c11 = c11 + a1 * b1; c21 = c21 + a2 * b1; c31 = c31 + a3 * b1
                c02 = c02 + a0 * b2; c12 = c12 + a1 * b2; c22 = c22 + a2 * b2; c32 = c32 + a3 * b2
                c03 = c03 + a0 * b3; c13 = c13 + a1 * b3; c23 = c23 + a2 * b3; c33 = c33 + a3 * b3
              end do
              C(i,         j)         = C(i,         j)         + alpha * c00
              C(i + 1_i32, j)         = C(i + 1_i32, j)         + alpha * c10
              C(i + 2_i32, j)         = C(i + 2_i32, j)         + alpha * c20
              C(i + 3_i32, j)         = C(i + 3_i32, j)         + alpha * c30
              C(i,         j + 1_i32) = C(i,         j + 1_i32) + alpha * c01
              C(i + 1_i32, j + 1_i32) = C(i + 1_i32, j + 1_i32) + alpha * c11
              C(i + 2_i32, j + 1_i32) = C(i + 2_i32, j + 1_i32) + alpha * c21
              C(i + 3_i32, j + 1_i32) = C(i + 3_i32, j + 1_i32) + alpha * c31
              C(i,         j + 2_i32) = C(i,         j + 2_i32) + alpha * c02
              C(i + 1_i32, j + 2_i32) = C(i + 1_i32, j + 2_i32) + alpha * c12
              C(i + 2_i32, j + 2_i32) = C(i + 2_i32, j + 2_i32) + alpha * c22
              C(i + 3_i32, j + 2_i32) = C(i + 3_i32, j + 2_i32) + alpha * c32
              C(i,         j + 3_i32) = C(i,         j + 3_i32) + alpha * c03
              C(i + 1_i32, j + 3_i32) = C(i + 1_i32, j + 3_i32) + alpha * c13
              C(i + 2_i32, j + 3_i32) = C(i + 2_i32, j + 3_i32) + alpha * c23
              C(i + 3_i32, j + 3_i32) = C(i + 3_i32, j + 3_i32) + alpha * c33
            end do

            do i = i_lim4 + 1_i32, i_end
              c00 = 0.0_dp; c01 = 0.0_dp; c02 = 0.0_dp; c03 = 0.0_dp
              do l = l0, l_end
                a0  = A(l, i)
                c00 = c00 + a0 * B(l, j)
                c01 = c01 + a0 * B(l, j + 1_i32)
                c02 = c02 + a0 * B(l, j + 2_i32)
                c03 = c03 + a0 * B(l, j + 3_i32)
              end do
              C(i, j)         = C(i, j)         + alpha * c00
              C(i, j + 1_i32) = C(i, j + 1_i32) + alpha * c01
              C(i, j + 2_i32) = C(i, j + 2_i32) + alpha * c02
              C(i, j + 3_i32) = C(i, j + 3_i32) + alpha * c03
            end do
          end do

          do j = j_lim4 + 1_i32, j_end
            do i = i0, i_lim4, 4_i32
              c00 = 0.0_dp; c10 = 0.0_dp; c20 = 0.0_dp; c30 = 0.0_dp
              do l = l0, l_end
                b0  = B(l, j)
                c00 = c00 + A(l, i)         * b0
                c10 = c10 + A(l, i + 1_i32) * b0
                c20 = c20 + A(l, i + 2_i32) * b0
                c30 = c30 + A(l, i + 3_i32) * b0
              end do
              C(i,         j) = C(i,         j) + alpha * c00
              C(i + 1_i32, j) = C(i + 1_i32, j) + alpha * c10
              C(i + 2_i32, j) = C(i + 2_i32, j) + alpha * c20
              C(i + 3_i32, j) = C(i + 3_i32, j) + alpha * c30
            end do

            do i = i_lim4 + 1_i32, i_end
              c00 = 0.0_dp
              do l = l0, l_end
                c00 = c00 + A(l, i) * B(l, j)
              end do
              C(i, j) = C(i, j) + alpha * c00
            end do
          end do
        end do
      end do
    end do
  end subroutine fc_gemm_tn

  !---------------------------------------------------------------------------
  ! C(1:m, 1:n) := alpha * A(1:m, 1:k) * B(1:n, 1:k)^T + beta * C(1:m, 1:n)
  ! L1/L2 cache-tiled (MC=128, NC=64, KC=128) with 1x4 column-unrolled SIMD
  ! FMA register micro-kernel and unit-stride i inner loop (0 stack arrays).
  !---------------------------------------------------------------------------
  pure subroutine fc_gemm_nt(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    integer(i32), intent(in) :: m, n, k, lda, ldb, ldc
    real(dp), intent(in)     :: alpha, beta
    real(dp), intent(in)     :: A(lda, *), B(ldb, *)
    real(dp), intent(inout)  :: C(ldc, *)

    integer(i32), parameter :: MC = 128_i32, NC = 64_i32, KC = 128_i32
    integer(i32) :: i, j, l, i0, j0, l0, i_end, j_end, l_end, j_lim4
    real(dp)     :: b0, b1, b2, b3, a_il

    if (m <= 0_i32 .or. n <= 0_i32) return

#ifdef FORTCHUCK_USE_BLAS
    if (int(m, i64) * int(n, i64) * int(max(k, 0_i32), i64) > 8192_i64) then
      call dgemm('N', 'T', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
      return
    end if
#endif

    if (.not. (abs(beta) > 0.0_dp)) then
      do j = 1_i32, n
        do i = 1_i32, m
          C(i, j) = 0.0_dp
        end do
      end do
    else if (beta < 1.0_dp .or. beta > 1.0_dp) then
      do j = 1_i32, n
        do i = 1_i32, m
          C(i, j) = beta * C(i, j)
        end do
      end do
    end if

    if (k <= 0_i32 .or. .not. (abs(alpha) > 0.0_dp)) return

    do j0 = 1_i32, n, NC
      j_end  = min(j0 + NC - 1_i32, n)
      j_lim4 = j0 + ((j_end - j0 + 1_i32) / 4_i32) * 4_i32 - 1_i32
      do l0 = 1_i32, k, KC
        l_end = min(l0 + KC - 1_i32, k)
        do i0 = 1_i32, m, MC
          i_end = min(i0 + MC - 1_i32, m)

          do j = j0, j_lim4, 4_i32
            do l = l0, l_end
              b0 = alpha * B(j,         l)
              b1 = alpha * B(j + 1_i32, l)
              b2 = alpha * B(j + 2_i32, l)
              b3 = alpha * B(j + 3_i32, l)
              do i = i0, i_end
                a_il            = A(i, l)
                C(i, j)         = C(i, j)         + a_il * b0
                C(i, j + 1_i32) = C(i, j + 1_i32) + a_il * b1
                C(i, j + 2_i32) = C(i, j + 2_i32) + a_il * b2
                C(i, j + 3_i32) = C(i, j + 3_i32) + a_il * b3
              end do
            end do
          end do

          do j = j_lim4 + 1_i32, j_end
            do l = l0, l_end
              b0 = alpha * B(j, l)
              if (abs(b0) > 0.0_dp) then
                do i = i0, i_end
                  C(i, j) = C(i, j) + A(i, l) * b0
                end do
              end if
            end do
          end do
        end do
      end do
    end do
  end subroutine fc_gemm_nt

  !---------------------------------------------------------------------------
  ! Symmetric rank-k product S = X^T * X in R^{p x p} with exact symmetry.
  !---------------------------------------------------------------------------
  pure subroutine fc_syrk_tn(n, p, X, S)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X(n, p)
    real(dp), intent(out)    :: S(p, p)

    integer(i32) :: i, j, l
    real(dp)     :: acc

    if (p <= 0_i32) return

    do j = 1_i32, p
      do i = 1_i32, j
        acc = 0.0_dp
        do l = 1_i32, n
          acc = acc + X(l, i) * X(l, j)
        end do
        S(i, j) = acc
        if (i < j) S(j, i) = acc
      end do
    end do
  end subroutine fc_syrk_tn

  !---------------------------------------------------------------------------
  ! Frobenius inner product <A, B>_F = tr(A^T B).
  !---------------------------------------------------------------------------
  pure function fc_frob_inner(m, n, A, B) result(val)
    integer(i32), intent(in) :: m, n
    real(dp), intent(in)     :: A(m, n), B(m, n)
    real(dp)                 :: val

    integer(i32) :: i, j

    val = 0.0_dp
    if (m <= 0_i32 .or. n <= 0_i32) return

    do j = 1_i32, n
      do i = 1_i32, m
        val = val + A(i, j) * B(i, j)
      end do
    end do
  end function fc_frob_inner

  !---------------------------------------------------------------------------
  ! Overflow-safe Frobenius norm ||A||_F via DLASSQ scaled sum-of-squares.
  !---------------------------------------------------------------------------
  pure function fc_frob_norm(m, n, A) result(val)
    integer(i32), intent(in) :: m, n
    real(dp), intent(in)     :: A(m, n)
    real(dp)                 :: val

    integer(i32) :: i, j
    real(dp)     :: scale, ssq, absxi, ratio

    val = 0.0_dp
    if (m <= 0_i32 .or. n <= 0_i32) return

    scale = 0.0_dp
    ssq   = 1.0_dp
    do j = 1_i32, n
      do i = 1_i32, m
        if (ieee_is_nan(A(i, j))) then
          val = A(i, j)
          return
        end if
        absxi = abs(A(i, j))
        if (absxi > 0.0_dp) then
          if (.not. ieee_is_finite(absxi)) then
            val = absxi
            return
          end if
          if (scale < absxi) then
            ratio = scale / absxi
            ssq   = 1.0_dp + ssq * (ratio * ratio)
            scale = absxi
          else
            ratio = absxi / scale
            ssq   = ssq + ratio * ratio
          end if
        end if
      end do
    end do
    val = scale * sqrt(ssq)
  end function fc_frob_norm

  !---------------------------------------------------------------------------
  ! Overflow-safe hypot(a, b) = sqrt(a^2 + b^2) without FE_OVERFLOW.
  !---------------------------------------------------------------------------
  pure function safe_hypot(a, b) result(r)
    real(dp), intent(in) :: a, b
    real(dp)             :: r
    real(dp)             :: x, y, mx, mn, q

    if (ieee_is_nan(a)) then
      r = a
      return
    end if
    if (ieee_is_nan(b)) then
      r = b
      return
    end if
    x  = abs(a)
    y  = abs(b)
    mx = max(x, y)
    mn = min(x, y)
    if (.not. (mx > 0.0_dp)) then
      r = 0.0_dp
    else if (.not. ieee_is_finite(mx)) then
      r = mx
    else
      q = mn / mx
      r = mx * sqrt(1.0_dp + q * q)
    end if
  end function safe_hypot

  !---------------------------------------------------------------------------
  ! In-place partial-pivoted P * A = L * U with IEEE NaN/Inf check and
  ! reciprocal pivot guard min_i |U_ii| < piv_tol * max_i |U_ii|.
  !---------------------------------------------------------------------------
  pure subroutine fc_lu_factor(n, A, ipiv, piv_tol, info)
    integer(i32), intent(in)  :: n
    real(dp), intent(inout)   :: A(n, n)
    integer(i32), intent(out) :: ipiv(n)
    real(dp), intent(in)      :: piv_tol
    integer(i32), intent(out) :: info

    integer(i32) :: i, j, k, piv
    real(dp)     :: piv_abs, cand_abs, u_max, tmp, a_kj, tol_eff

    info = 0_i32
    if (n <= 0_i32) return

    do k = 1_i32, n
      ipiv(k) = k
    end do

    tol_eff = max(0.0_dp, piv_tol)

    do j = 1_i32, n
      do i = 1_i32, n
        if (ieee_is_nan(A(i, j)) .or. .not. ieee_is_finite(A(i, j))) then
          info = 1_i32
          return
        end if
      end do
    end do

    u_max = 0.0_dp
    do k = 1_i32, n
      piv     = k
      piv_abs = abs(A(k, k))
      do i = k + 1_i32, n
        cand_abs = abs(A(i, k))
        if (ieee_is_nan(cand_abs)) then
          info = k
          return
        end if
        if (cand_abs > piv_abs) then
          piv_abs = cand_abs
          piv     = i
        end if
      end do

      ipiv(k) = piv
      if (.not. ieee_is_finite(piv_abs) .or. piv_abs <= tiny(1.0_dp)) then
        info = k
        return
      end if

      if (piv /= k) then
        do j = 1_i32, n
          tmp       = A(k, j)
          A(k, j)   = A(piv, j)
          A(piv, j) = tmp
        end do
      end if

      if (piv_abs > u_max) u_max = piv_abs
      if (piv_abs < tol_eff * u_max) then
        info = k
        return
      end if

      do i = k + 1_i32, n
        A(i, k) = A(i, k) / A(k, k)
      end do

      do j = k + 1_i32, n
        a_kj = A(k, j)
        if (abs(a_kj) > 0.0_dp) then
          do i = k + 1_i32, n
            A(i, j) = A(i, j) - A(i, k) * a_kj
          end do
        end if
      end do
    end do

    ! Final global reciprocal pivot ratio verification across all U(i, i)
    do i = 1_i32, n
      if (ieee_is_nan(A(i, i)) .or. .not. ieee_is_finite(A(i, i))) then
        info = i
        return
      end if
      if (abs(A(i, i)) < tol_eff * u_max .or. abs(A(i, i)) <= tiny(1.0_dp)) then
        info = i
        return
      end if
    end do
  end subroutine fc_lu_factor

  !---------------------------------------------------------------------------
  ! Solve A * X = B in-place on B(n, nrhs) given P * A = L * U.
  ! Hoists diagonal reciprocals inv_ukk = 1 / LU(k,k) outside RHS loops and
  ! processes RHS columns in JB = 4 register panels (0 stack arrays).
  !---------------------------------------------------------------------------
  pure subroutine fc_lu_solve(n, nrhs, LU, ipiv, B)
    integer(i32), intent(in) :: n, nrhs
    real(dp), intent(in)     :: LU(n, n)
    integer(i32), intent(in) :: ipiv(n)
    real(dp), intent(inout)  :: B(n, nrhs)

    integer(i32) :: i, j, k, piv, nrhs_lim4
    real(dp)     :: tmp, inv_ukk, l_ik, u_ik, b0, b1, b2, b3

    if (n <= 0_i32 .or. nrhs <= 0_i32) return

    nrhs_lim4 = (nrhs / 4_i32) * 4_i32

    ! 1. Apply row permutations P * B
    do k = 1_i32, n
      piv = ipiv(k)
      if (piv /= k) then
        do j = 1_i32, nrhs_lim4, 4_i32
          tmp             = B(k,   j)
          B(k,   j)       = B(piv, j)
          B(piv, j)       = tmp
          tmp             = B(k,   j + 1_i32)
          B(k,   j + 1_i32) = B(piv, j + 1_i32)
          B(piv, j + 1_i32) = tmp
          tmp             = B(k,   j + 2_i32)
          B(k,   j + 2_i32) = B(piv, j + 2_i32)
          B(piv, j + 2_i32) = tmp
          tmp             = B(k,   j + 3_i32)
          B(k,   j + 3_i32) = B(piv, j + 3_i32)
          B(piv, j + 3_i32) = tmp
        end do
        do j = nrhs_lim4 + 1_i32, nrhs
          tmp       = B(k, j)
          B(k, j)   = B(piv, j)
          B(piv, j) = tmp
        end do
      end if
    end do

    ! 2. Forward substitution L * Y = P * B in JB = 4 RHS panels + scalar tail
    do j = 1_i32, nrhs_lim4, 4_i32
      do k = 1_i32, n - 1_i32
        b0 = B(k, j)
        b1 = B(k, j + 1_i32)
        b2 = B(k, j + 2_i32)
        b3 = B(k, j + 3_i32)
        do i = k + 1_i32, n
          l_ik            = LU(i, k)
          B(i, j)         = B(i, j)         - l_ik * b0
          B(i, j + 1_i32) = B(i, j + 1_i32) - l_ik * b1
          B(i, j + 2_i32) = B(i, j + 2_i32) - l_ik * b2
          B(i, j + 3_i32) = B(i, j + 3_i32) - l_ik * b3
        end do
      end do
    end do

    do j = nrhs_lim4 + 1_i32, nrhs
      do k = 1_i32, n - 1_i32
        b0 = B(k, j)
        if (abs(b0) > 0.0_dp) then
          do i = k + 1_i32, n
            B(i, j) = B(i, j) - LU(i, k) * b0
          end do
        end if
      end do
    end do

    ! 3. Back substitution U * X = Y with hoisted inv_ukk + JB = 4 RHS panels
    do k = n, 1_i32, -1_i32
      inv_ukk = 1.0_dp / LU(k, k)
      do j = 1_i32, nrhs_lim4, 4_i32
        b0              = B(k, j)         * inv_ukk
        b1              = B(k, j + 1_i32) * inv_ukk
        b2              = B(k, j + 2_i32) * inv_ukk
        b3              = B(k, j + 3_i32) * inv_ukk
        B(k, j)         = b0
        B(k, j + 1_i32) = b1
        B(k, j + 2_i32) = b2
        B(k, j + 3_i32) = b3
        do i = 1_i32, k - 1_i32
          u_ik            = LU(i, k)
          B(i, j)         = B(i, j)         - u_ik * b0
          B(i, j + 1_i32) = B(i, j + 1_i32) - u_ik * b1
          B(i, j + 2_i32) = B(i, j + 2_i32) - u_ik * b2
          B(i, j + 3_i32) = B(i, j + 3_i32) - u_ik * b3
        end do
      end do

      do j = nrhs_lim4 + 1_i32, nrhs
        b0      = B(k, j) * inv_ukk
        B(k, j) = b0
        if (abs(b0) > 0.0_dp) then
          do i = 1_i32, k - 1_i32
            B(i, j) = B(i, j) - LU(i, k) * b0
          end do
        end if
      end do
    end do
  end subroutine fc_lu_solve

  !---------------------------------------------------------------------------
  ! Solve A^T * X = B in-place on B(n, nrhs) given P * A = L * U
  ! (A^T = U^T * L^T * P), using hoisted inv_uii = 1 / LU(i,i) and JB = 4
  ! RHS register panels over unit-stride k inner loops (0 stack arrays).
  !---------------------------------------------------------------------------
  pure subroutine fc_lu_solve_trans(n, nrhs, LU, ipiv, B)
    integer(i32), intent(in) :: n, nrhs
    real(dp), intent(in)     :: LU(n, n)
    integer(i32), intent(in) :: ipiv(n)
    real(dp), intent(inout)  :: B(n, nrhs)

    integer(i32) :: i, j, k, piv, nrhs_lim4
    real(dp)     :: tmp, inv_uii, u_ki, l_ki, acc0, acc1, acc2, acc3

    if (n <= 0_i32 .or. nrhs <= 0_i32) return

    nrhs_lim4 = (nrhs / 4_i32) * 4_i32

    ! 1. Solve U^T * Y = B with hoisted inv_uii = 1 / LU(i,i) across all RHS
    do i = 1_i32, n
      inv_uii = 1.0_dp / LU(i, i)
      do j = 1_i32, nrhs_lim4, 4_i32
        acc0 = B(i, j)
        acc1 = B(i, j + 1_i32)
        acc2 = B(i, j + 2_i32)
        acc3 = B(i, j + 3_i32)
        do k = 1_i32, i - 1_i32
          u_ki = LU(k, i)
          acc0 = acc0 - u_ki * B(k, j)
          acc1 = acc1 - u_ki * B(k, j + 1_i32)
          acc2 = acc2 - u_ki * B(k, j + 2_i32)
          acc3 = acc3 - u_ki * B(k, j + 3_i32)
        end do
        B(i, j)         = acc0 * inv_uii
        B(i, j + 1_i32) = acc1 * inv_uii
        B(i, j + 2_i32) = acc2 * inv_uii
        B(i, j + 3_i32) = acc3 * inv_uii
      end do

      do j = nrhs_lim4 + 1_i32, nrhs
        acc0 = B(i, j)
        do k = 1_i32, i - 1_i32
          acc0 = acc0 - LU(k, i) * B(k, j)
        end do
        B(i, j) = acc0 * inv_uii
      end do
    end do

    ! 2. Solve L^T * Z = Y in JB = 4 RHS panels + scalar tail
    do j = 1_i32, nrhs_lim4, 4_i32
      do i = n - 1_i32, 1_i32, -1_i32
        acc0 = B(i, j)
        acc1 = B(i, j + 1_i32)
        acc2 = B(i, j + 2_i32)
        acc3 = B(i, j + 3_i32)
        do k = i + 1_i32, n
          l_ki = LU(k, i)
          acc0 = acc0 - l_ki * B(k, j)
          acc1 = acc1 - l_ki * B(k, j + 1_i32)
          acc2 = acc2 - l_ki * B(k, j + 2_i32)
          acc3 = acc3 - l_ki * B(k, j + 3_i32)
        end do
        B(i, j)         = acc0
        B(i, j + 1_i32) = acc1
        B(i, j + 2_i32) = acc2
        B(i, j + 3_i32) = acc3
      end do
    end do

    do j = nrhs_lim4 + 1_i32, nrhs
      do i = n - 1_i32, 1_i32, -1_i32
        acc0 = B(i, j)
        do k = i + 1_i32, n
          acc0 = acc0 - LU(k, i) * B(k, j)
        end do
        B(i, j) = acc0
      end do
    end do

    ! 3. Apply P^T: reverse swap sequence k = n .. 1
    do k = n, 1_i32, -1_i32
      piv = ipiv(k)
      if (piv /= k) then
        do j = 1_i32, nrhs_lim4, 4_i32
          tmp               = B(k,   j)
          B(k,   j)         = B(piv, j)
          B(piv, j)         = tmp
          tmp               = B(k,   j + 1_i32)
          B(k,   j + 1_i32) = B(piv, j + 1_i32)
          B(piv, j + 1_i32) = tmp
          tmp               = B(k,   j + 2_i32)
          B(k,   j + 2_i32) = B(piv, j + 2_i32)
          B(piv, j + 2_i32) = tmp
          tmp               = B(k,   j + 3_i32)
          B(k,   j + 3_i32) = B(piv, j + 3_i32)
          B(piv, j + 3_i32) = tmp
        end do
        do j = nrhs_lim4 + 1_i32, nrhs
          tmp       = B(k, j)
          B(k, j)   = B(piv, j)
          B(piv, j) = tmp
        end do
      end if
    end do
  end subroutine fc_lu_solve_trans

  !---------------------------------------------------------------------------
  ! Solve Z * J = Panel in-place on Panel(m, p) via unit-stride column sweeps,
  ! given LU_Jt and ipiv from fc_lu_factor(p, J^T, ipiv, ...).
  ! Since P * J^T = L * U, J = U^T * L^T * P, so Z * U^T * L^T = Panel * P^T.
  !---------------------------------------------------------------------------
  pure subroutine fc_lu_solve_right(m, p, LU_Jt, ipiv, Panel)
    integer(i32), intent(in) :: m, p
    real(dp), intent(in)     :: LU_Jt(p, p)
    integer(i32), intent(in) :: ipiv(p)
    real(dp), intent(inout)  :: Panel(m, p)

    integer(i32) :: i, j, k, piv
    real(dp)     :: tmp, l_jk, u_kj, inv_ujj

    if (m <= 0_i32 .or. p <= 0_i32) return

    ! 1. Form B1 = Panel * P^T (equivalent to B1^T = P * Panel^T: swap cols k, ipiv(k))
    do k = 1_i32, p
      piv = ipiv(k)
      if (piv /= k) then
        do i = 1_i32, m
          tmp           = Panel(i, k)
          Panel(i, k)   = Panel(i, piv)
          Panel(i, piv) = tmp
        end do
      end if
    end do

    ! 2. Solve Y * L^T = B1 (unit-stride column AXPYs)
    do k = 1_i32, p - 1_i32
      do j = k + 1_i32, p
        l_jk = LU_Jt(j, k)
        if (abs(l_jk) > 0.0_dp) then
          do i = 1_i32, m
            Panel(i, j) = Panel(i, j) - Panel(i, k) * l_jk
          end do
        end if
      end do
    end do

    ! 3. Solve Z * U^T = Y backward j = p .. 1 (unit-stride column scale + AXPYs)
    do j = p, 1_i32, -1_i32
      inv_ujj = 1.0_dp / LU_Jt(j, j)
      do i = 1_i32, m
        Panel(i, j) = Panel(i, j) * inv_ujj
      end do
      do k = 1_i32, j - 1_i32
        u_kj = LU_Jt(k, j)
        if (abs(u_kj) > 0.0_dp) then
          do i = 1_i32, m
            Panel(i, k) = Panel(i, k) - Panel(i, j) * u_kj
          end do
        end if
      end do
    end do
  end subroutine fc_lu_solve_right

  !---------------------------------------------------------------------------
  ! In-place thin Householder QR replacing X in R^{n x p} with Q in St(n, p)
  ! such that X_orig = Q * R with R(i, i) > 0 for all i = 1..p.
  ! Sets info = i and returns early if X is non-finite or |R(i, i)| <= 1e-14.
  !---------------------------------------------------------------------------
  pure subroutine fc_geqrf_pos(n, p, X, tau_qr, work_qr, info)
    integer(i32), intent(in)  :: n, p
    real(dp), intent(inout)   :: X(n, p)
    real(dp), intent(out)     :: tau_qr(p)
    real(dp), intent(out)     :: work_qr(p)
    integer(i32), intent(out) :: info

    integer(i32) :: i, j, r
    real(dp)     :: alpha, xnorm, r_norm, beta_hh, v_scale, tau_i, w_j, s_i, r_ij, inv_nrm

    info = 0_i32
    if (p <= 0_i32) return

    do i = 1_i32, p
      tau_qr(i)  = 0.0_dp
      work_qr(i) = 0.0_dp
    end do

    if (n < p) then
      info = 1_i32
      return
    end if

    do j = 1_i32, p
      do r = 1_i32, n
        if (ieee_is_nan(X(r, j)) .or. .not. ieee_is_finite(X(r, j))) then
          info = 1_i32
          return
        end if
      end do
    end do

    ! Phase 1: Cancellation-free Householder QR factorization X = Q_std * R_std
    do i = 1_i32, p
      alpha = X(i, i)
      if (i < n) then
        xnorm = fc_frob_norm(n - i, 1_i32, X(i + 1_i32:n, i))
      else
        xnorm = 0.0_dp
      end if

      if (.not. (xnorm > 0.0_dp)) then
        if (abs(alpha) <= 1.0e-14_dp) then
          info = i
          return
        end if
        tau_qr(i) = 0.0_dp
      else
        r_norm = safe_hypot(alpha, xnorm)
        if (.not. ieee_is_finite(r_norm) .or. r_norm <= 1.0e-14_dp) then
          info = i
          return
        end if
        if (alpha >= 0.0_dp) then
          beta_hh = -r_norm
        else
          beta_hh = r_norm
        end if
        tau_i     = (beta_hh - alpha) / beta_hh
        v_scale   = 1.0_dp / (alpha - beta_hh)
        tau_qr(i) = tau_i
        X(i, i)   = beta_hh
        do r = i + 1_i32, n
          X(r, i) = X(r, i) * v_scale
        end do

        ! Apply H_i = I - tau_i * v_i * v_i^T to trailing columns j = i+1 .. p
        do j = i + 1_i32, p
          w_j = X(i, j)
          do r = i + 1_i32, n
            w_j = w_j + X(r, i) * X(r, j)
          end do
          w_j       = tau_i * w_j
          work_qr(j) = w_j
          X(i, j)   = X(i, j) - w_j
          do r = i + 1_i32, n
            X(r, j) = X(r, j) - w_j * X(r, i)
          end do
        end do
      end if
    end do

    ! Phase 2: Backward accumulation of Q_+ = Q_std * diag(sign(R_std_ii))
    ! so that R_+ = diag(sign(R_std_ii)) * R_std has strictly positive diagonal.
    do i = p, 1_i32, -1_i32
      if (X(i, i) >= 0.0_dp) then
        s_i = 1.0_dp
      else
        s_i = -1.0_dp
      end if
      tau_i = tau_qr(i)

      if (abs(tau_i) > 0.0_dp) then
        do j = i + 1_i32, p
          w_j = X(i, j)
          do r = i + 1_i32, n
            w_j = w_j + X(r, i) * X(r, j)
          end do
          w_j        = tau_i * w_j
          work_qr(j) = w_j
          X(i, j)    = X(i, j) - w_j
          do r = i + 1_i32, n
            X(r, j) = X(r, j) - w_j * X(r, i)
          end do
        end do
      end if

      do r = 1_i32, i - 1_i32
        X(r, i) = 0.0_dp
      end do
      X(i, i) = s_i * (1.0_dp - tau_i)
      if (abs(tau_i) > 0.0_dp) then
        do r = i + 1_i32, n
          X(r, i) = (-s_i * tau_i) * X(r, i)
        end do
      else
        do r = i + 1_i32, n
          X(r, i) = 0.0_dp
        end do
      end if
    end do

    ! Phase 3: Single upper-triangular MGS polish (preserves R_ii > 0 and
    ! drives ||Q^T Q - I_p||_F down to machine-epsilon ~ 2e-16 < 1e-15).
    do j = 1_i32, p
      do i = 1_i32, j - 1_i32
        r_ij = 0.0_dp
        do r = 1_i32, n
          r_ij = r_ij + X(r, i) * X(r, j)
        end do
        if (abs(r_ij) > 0.0_dp) then
          do r = 1_i32, n
            X(r, j) = X(r, j) - r_ij * X(r, i)
          end do
        end if
      end do
      r_norm = fc_frob_norm(n, 1_i32, X(1_i32:n, j))
      if (r_norm > 0.0_dp) then
        inv_nrm = 1.0_dp / r_norm
        do r = 1_i32, n
          X(r, j) = X(r, j) * inv_nrm
        end do
      end if
    end do
  end subroutine fc_geqrf_pos

end module fortchuck_linalg
