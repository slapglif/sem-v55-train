import Lake
open Lake DSL

package sem where
  leanOptions := #[
    ⟨`autoImplicit, .ofBool false⟩
  ]

@[default_target]
lean_lib SEM where
  roots := #[`SEM]
