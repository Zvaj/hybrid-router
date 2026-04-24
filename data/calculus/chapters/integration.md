# Integration

## Antiderivatives and the Indefinite Integral

Integration is the reverse of differentiation. Given a function f(x), an Antiderivative F(x) satisfies F′(x) = f(x). The Indefinite Integral ∫f(x)dx denotes the family of all antiderivatives: F(x) + C, where C is the Constant of Integration representing the unknown vertical shift. Integral Notation uses the elongated S to evoke summation, and the Integrand is the function being integrated.

The Power Rule Integration mirrors the derivative rule: ∫xⁿdx = xⁿ⁺¹/(n+1) + C for n ≠ −1. The Constant Rule Integr, Sum Rule Integration, and Difference Rule Integr let you integrate term by term. Basic Trig Integrals are fixed: ∫sin x dx = −cos x + C, ∫cos x dx = sin x + C. The Integral of e to x is e^x + C; the Integral of 1 Over x is ln|x| + C.

## Definite Integrals and the Riemann Sum

The Definite Integral ∫ₐᵇ f(x)dx gives the Net Signed Area between a curve and the x-axis from a to b. Positive area accumulates above the axis; negative area below. The Riemann Sum approximates this area by dividing the interval into subintervals and summing rectangle areas. The Left Riemann Sum uses left endpoints, the Right Riemann Sum uses right endpoints, and the Midpoint Riemann Sum uses midpoints. The Trapezoidal Rule averages left and right sums for better accuracy. The Definite Integral is the Limit of Riemann Sum as the number of rectangles grows to infinity.

Definite integrals satisfy several Integral Properties: the integral of a sum splits, reversing the Limits of Integration negates the result, and the Zero Width Integral (same upper and lower bound) equals zero.

## The Fundamental Theorem

The Fundamental Theorem of Calculus unifies differentiation and integration. FTC Part One states that if F(x) = ∫ₐˣ f(t)dt, then F′(x) = f(x) — the Derivative of Integral recovers the integrand. FTC Part Two (the Evaluation Theorem) provides the computational workhorse: ∫ₐᵇ f(x)dx = F(b) − F(a), where F is any antiderivative of f. The Accumulation Function F(x) = ∫ₐˣ f(t)dt tracks how area accumulates, and the Net Change Theorem interprets the definite integral as total accumulated change.

## Substitution

u-Substitution is the integration technique dual to the Chain Rule. The Substitution Method replaces a complicated expression with a single variable u. Choosing u to be the inside function of a composition, computing the du Calculation from the derivative, and rewriting everything in terms of u transforms the integral into a simpler form. After integrating in u, Back Substitution returns the result in terms of x. For definite integrals, Changing Bounds to match u avoids back-substitution entirely.

The Integration Strategy for choosing techniques starts with algebraic simplification, tries u-Substitution first, and escalates to Long Division Method (for rational functions with numerator degree ≥ denominator) or Partial Fractions (for factorable denominators).
