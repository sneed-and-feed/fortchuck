import Lake
open Lake DSL

package "formalization" where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "de2ef68216c6074f338c8e61890ee0a379ddfb9b"

@[default_target]
lean_lib «Fortchuck» where
  srcDir := "."
