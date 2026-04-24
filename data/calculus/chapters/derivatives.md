# Derivatives

## The Formal Definition

A derivative measures the instantaneous rate of change of a function at a point. Formally, the Derivative Definition is the limit of the difference quotient: f′(x) = lim(h→0) [f(x+h) − f(x)] / h. This expression, called the Limit Definition Deriv, captures the slope of the tangent line at a point by taking the Secant Line slope and shrinking the interval to zero. When this limit exists, the function is differentiable at that point. The Derivative at a Point is the numerical value of this limit for a specific x; the Derivative Function f′ is what you get by treating x as a variable.

Geometrically, the Tangent Line at a point touches the curve locally and has slope equal to the derivative. This connects Derivative Notation — f′(x), dy/dx, and Leibniz Notation — to a concrete geometric object: the slope of the Tangent Line.

## Derivative Rules

Rather than computing limits each time, Derivative Rules provide shortcuts. The most fundamental is the Power Rule Derivative: d/dx[xⁿ] = nxⁿ⁻¹. Combined with the Constant Rule (derivative of a constant is zero), Sum Rule Derivative, and Difference Rule Deriv, these handle any Polynomial Derivative in seconds. The Constant Multiple Deriv rule lets constants factor out unchanged.

For functions built from two parts, the Product Rule applies when multiplying: d/dx[f·g] = f′g + fg′. The Quotient Rule handles division: d/dx[f/g] = (f′g − fg′)/g². These rules require knowing what each piece does separately before combining.

Trigonometric derivatives follow fixed patterns: the Derivative of Sine is cosine, and the Derivative of Cosine is negative sine. Exponential functions have an elegant property — the Derivative of e to x is itself, e^x. The Derivative of ln x is 1/x.

## The Chain Rule

The Chain Rule is the most important rule for Composite Function differentiation. If y = f(g(x)), then dy/dx = f′(g(x)) · g′(x). The Chain Rule Formula says: differentiate the outside function leaving the inside alone, then multiply by the derivative of the inside. Recognizing which part is the Inside Function and which is the Outside Function is the core skill. The Power Chain Rule handles cases like (sin x)⁵; the Exponential Chain Rule handles e^(g(x)); the Log Chain Rule handles ln(g(x)).

## Implicit Differentiation

When a relationship between x and y is not written explicitly as y = f(x), Implicit Differentiation applies. Treating y as Function of x and differentiating both sides, every y term picks up a dy/dx factor via the Implicit Chain Rule. Solving for dy/dx gives the slope at any point on the Implicit Equation's curve.

## Higher-Order Derivatives

The Second Derivative f″(x) is the derivative of f′(x). It measures concavity — whether the function curves upward or downward. The Third Derivative and beyond (nth Derivative) appear in physics (Acceleration as the second derivative of position) and in error analysis.
