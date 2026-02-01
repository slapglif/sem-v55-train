/-
  SEM V5.5 "Lean Crystal" - Basic Types

  Defines the foundational types for the Signal-Entropic Model:
  - WaveFunction: Complex-valued state vector
  - Token: Discrete vocabulary element
  - GlobalState: The full system state
  - Time: Continuous time parameter
-/

-- We work in a simplified framework suitable for the axioms
-- Full Mathlib integration would be needed for complete proofs

namespace SEM

/-- Complex number representation -/
structure Complex where
  re : Float
  im : Float
  deriving Repr, BEq

namespace Complex

def zero : Complex := ⟨0.0, 0.0⟩
def one : Complex := ⟨1.0, 0.0⟩
def i : Complex := ⟨0.0, 1.0⟩

def add (a b : Complex) : Complex :=
  ⟨a.re + b.re, a.im + b.im⟩

def mul (a b : Complex) : Complex :=
  ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩

def conj (a : Complex) : Complex :=
  ⟨a.re, -a.im⟩

def normSq (a : Complex) : Float :=
  a.re * a.re + a.im * a.im

def norm (a : Complex) : Float :=
  Float.sqrt (normSq a)

instance : Add Complex := ⟨add⟩
instance : Mul Complex := ⟨mul⟩
instance : Inhabited Complex := ⟨zero⟩

end Complex

/-- A WaveFunction is a vector of complex amplitudes -/
structure WaveFunction (n : Nat) where
  components : Fin n → Complex
  deriving Inhabited

/-- Token index into the vocabulary -/
def Token := Nat

/-- Global state of the system -/
structure GlobalState (n : Nat) where
  wavefunction : WaveFunction n
  deriving Inhabited

/-- Time parameter for evolution -/
def Time := Float

/-- L2 norm squared of a wavefunction: Σ|ψᵢ|² -/
def normSq {n : Nat} (ψ : WaveFunction n) : Float :=
  let rec go (k : Nat) (acc : Float) : Float :=
    if h : k < n then
      go (k + 1) (acc + Complex.normSq (ψ.components ⟨k, h⟩))
    else
      acc
  go 0 0.0

/-- L2 norm of a wavefunction -/
def wfNorm {n : Nat} (ψ : WaveFunction n) : Float :=
  Float.sqrt (normSq ψ)

/-- Inner product ⟨φ|ψ⟩ = Σ conj(φᵢ) * ψᵢ -/
def innerProduct {n : Nat} (φ ψ : WaveFunction n) : Complex :=
  let rec go (k : Nat) (acc : Complex) : Complex :=
    if h : k < n then
      let term := Complex.mul (Complex.conj (φ.components ⟨k, h⟩)) (ψ.components ⟨k, h⟩)
      go (k + 1) (Complex.add acc term)
    else
      acc
  go 0 Complex.zero

end SEM
