
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
import ast
import io

st.set_page_config(
    page_title="Integration Visualizer",
    page_icon="📐",
    layout="centered"
)

x = sp.Symbol("x", real=True)

allowed = {
    "x": x,
    "pi": sp.pi,
    "e": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs
}

# Restricted mathematical expression parser
def parse(expression):
    if len(expression) > 150:
        raise ValueError("Expression is too long.")

    tree = ast.parse(
        expression.replace("^", "**"),
        mode="eval"
    )

    def convert(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                if abs(node.value) <= 1000000:
                    return sp.sympify(node.value)

        if isinstance(node, ast.Name):
            if node.id in ("x", "pi", "e"):
                return allowed[node.id]

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -convert(node.operand)
            if isinstance(node.op, ast.UAdd):
                return convert(node.operand)

        if isinstance(node, ast.BinOp):
            a = convert(node.left)
            b = convert(node.right)

            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                if b.is_number and abs(b) <= 20:
                    return a ** b

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in allowed and callable(allowed[name]):
                    if len(node.args) == 1 and not node.keywords:
                        return allowed[name](convert(node.args[0]))

        raise ValueError("Unsupported mathematical expression.")

    return convert(tree.body)


st.title("Integration Visualizer")

st.caption(
    "Explore integral calculus through "
    "interactive mathematical visualization."
)

with st.sidebar:
    st.header("About the Project")
    st.write("Mathematics Innovative Assignment")
    st.write("**K Hemant Kumar**")
    st.write("Registration No: 2602190045")
    st.write("Mechanical and Smart Manufacturing Engineering")
    st.write("Semester 1 | VSSUT, Burla")

    st.divider()

    st.header("Example Functions")
    st.code(
        "x^2\n"
        "sin(x)\n"
        "x^3-4*x\n"
        "exp(x)\n"
        "sqrt(x)\n"
        "1/(1+x^2)"
    )

with st.form("integration_form"):
    expression = st.text_input(
        "Enter mathematical function f(x)",
        value="sin(x)"
    )

    col1, col2 = st.columns(2)

    with col1:
        lower = st.text_input("Lower limit", "0")

    with col2:
        upper = st.text_input("Upper limit", "2*pi")

    submitted = st.form_submit_button(
        "Calculate and Visualize",
        use_container_width=True
    )

if submitted:
    try:
        expr = parse(expression)
        a_exact = parse(lower)
        b_exact = parse(upper)

        a = float(a_exact)
        b = float(b_exact)

        if not np.isfinite([a, b]).all():
            raise ValueError("Limits must be finite.")

        if a >= b:
            raise ValueError(
                "Upper limit must exceed lower limit."
            )

        if max(abs(a), abs(b)) > 1000:
            raise ValueError("Use limits between -1000 and 1000.")

        domain = sp.calculus.util.continuous_domain(
            expr, x, sp.S.Reals
        )

        if sp.Interval(a_exact, b_exact).is_subset(domain) != True:
            raise ValueError(
                "Function must be continuous over the interval."
            )

        f = sp.lambdify(x, expr, "numpy")

        result, error = quad(f, a, b)
        area, _ = quad(lambda t: abs(f(t)), a, b)

        antiderivative = sp.integrate(expr, x)

        st.subheader("Integration Results")

        st.latex(
            r"\int_{" + sp.latex(a_exact) + "}^{"
            + sp.latex(b_exact) + "}"
            + sp.latex(expr) + r"\,dx"
        )

        c1, c2 = st.columns(2)

        c1.metric("Definite Integral", f"{result:.6f}")
        c2.metric("Total Geometric Area", f"{area:.6f}")

        st.write("**Antiderivative**")

        if antiderivative.has(sp.Integral):
            st.info("Symbolic antiderivative unavailable.")
        else:
            st.latex(
                sp.latex(antiderivative) + "+C"
            )

        padding = (b - a) * 0.15

        X = np.linspace(
            a - padding, b + padding, 1500
        )

        XA = np.linspace(a, b, 1500)

        with np.errstate(all="ignore"):
            Y = np.broadcast_to(
                f(X), X.shape
            ).astype(float)

            YA = np.broadcast_to(
                f(XA), XA.shape
            ).astype(float)

        Y = np.where(np.isfinite(Y), Y, np.nan)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            X, Y, color="#2563eb",
            linewidth=2.5,
            label=f"$f(x)={sp.latex(expr)}$"
        )

        ax.fill_between(
            XA, YA, 0,
            where=YA >= 0,
            interpolate=True,
            color="#22c55e",
            alpha=0.4,
            label="Positive region"
        )

        ax.fill_between(
            XA, YA, 0,
            where=YA < 0,
            interpolate=True,
            color="#ef4444",
            alpha=0.4,
            label="Negative region"
        )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(a, color="gray", linestyle="--")
        ax.axvline(b, color="gray", linestyle="--")

        ax.set_title(
            "Definite Integration",
            fontsize=17,
            fontweight="bold"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(alpha=0.2)
        ax.legend()

        fig.tight_layout()

        st.pyplot(fig)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=300)

        st.download_button(
            "Download Graph",
            data=buffer.getvalue(),
            file_name="integration_graph.png",
            mime="image/png"
        )

        plt.close(fig)

        st.info(
            "The definite integral represents signed area, "
            "while total geometric area counts both positive "
            "and negative regions as positive."
        )

    except Exception as error:
        st.error(str(error))

st.divider()

st.caption(
    "Developed by K Hemant Kumar | "
    "Mathematics Innovative Project | 2026"
)
