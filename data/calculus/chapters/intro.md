# Introduction to Calculus

## The Derivative: Instantaneous Rate of Change

At its core, a derivative measures how fast something changes at a precise instant in time. Imagine you are driving a car and watching your speedometer — the needle shows not your average speed over the trip, but exactly how fast you are moving right now. That instantaneous reading is the intuition behind a derivative.

More formally, if you have a function f(x) that describes the position of an object at time x, then the derivative f'(x) gives you the velocity of that object at exactly the moment x. Suppose your position is described by f(t) = t², where t is time in seconds and f is measured in meters. At t = 3 seconds, the derivative f'(t) = 2t gives f'(3) = 6 meters per second. That is your exact speed at that instant — not an average over a span of time, but the instantaneous rate of change at a single point.

The derivative is defined through a limiting process: you look at the slope of the line connecting two nearby points on a curve, then slide those points together until they coincide. In that limit, the secant line becomes a tangent line, and its slope is the derivative.

## Limits: The Foundation of Calculus

A limit describes the value a function approaches as its input gets arbitrarily close to some point — without necessarily reaching it. We write lim(x→a) f(x) = L to say that f(x) gets as close to L as we like, provided x is close enough to a.

Limits are the bedrock on which everything else in calculus rests. Derivatives are defined as limits of difference quotients. Integrals are defined as limits of Riemann sums. Continuity is defined in terms of limits. Without a rigorous understanding of limits, none of the other machinery of calculus holds together. They allow mathematicians to reason carefully about behavior near a point — including points where a function may not even be defined — which is exactly the kind of precision that makes calculus such a powerful tool.

## The Chain Rule: Derivatives of Composite Functions

Many functions you encounter in practice are built by composing simpler ones. The chain rule tells you how to differentiate these compositions. If you have a function h(x) = f(g(x)), meaning you first apply g and then feed its output into f, then the derivative is:

h'(x) = f'(g(x)) · g'(x)

In plain English: differentiate the outer function evaluated at the inner function, then multiply by the derivative of the inner function.

For a concrete example, suppose h(x) = (3x + 1)⁵. Here g(x) = 3x + 1 and f(u) = u⁵. The derivative of f(u) = u⁵ is 5u⁴, so f'(g(x)) = 5(3x + 1)⁴. The derivative of g(x) = 3x + 1 is simply 3. Multiplying: h'(x) = 5(3x + 1)⁴ · 3 = 15(3x + 1)⁴. Without the chain rule, differentiating such composite functions would require laboriously expanding the power first.

## Integration: Area Under a Curve

Integration gives you a way to accumulate quantities. Geometrically, the definite integral of a function f(x) from a to b represents the area of the region bounded by the curve, the x-axis, and the vertical lines x = a and x = b.

The idea is built from a beautiful approximation: divide the interval [a, b] into many thin vertical strips. Each strip is approximately a rectangle whose height is the function value at that point and whose width is the tiny increment Δx. Summing all these rectangle areas gives an approximation of the total area. The definite integral is the limit of this sum as the rectangles are made infinitely thin and infinitely numerous — what mathematicians call a Riemann sum taken to its limit. The result captures the exact accumulated area, even when the function curves and the rectangles are only approximations.

## The Fundamental Theorem of Calculus

The most profound result in calculus is the fundamental theorem, which reveals that differentiation and integration are inverse operations. It has two parts. The first says that if you define a new function as the integral of f from a fixed point up to x, then the derivative of that new function is simply f(x) itself. The second part says that to evaluate a definite integral of f over [a, b], you only need to find an antiderivative F — a function whose derivative is f — and compute F(b) − F(a).

This is remarkable: computing areas, which seems like a geometric problem involving limits of sums, turns out to be equivalent to finding antiderivatives, which is an algebraic operation. The theorem unites the two central operations of calculus into a single coherent framework.
