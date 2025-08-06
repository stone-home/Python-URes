from typing import Optional
import warnings


# ============================================================================
# ENHANCED DEPENDENCY MANAGEMENT
# ============================================================================


def check_optional_dependency(
    package_name: str, import_name: Optional[str] = None
) -> bool:
    """
    Check if optional dependency is available

    Args:
            package_name: Name of the package to check
            import_name: Name to use for import (if different from package_name)

    Returns:
            bool: True if package is available, False otherwise
    """
    import_name = import_name or package_name

    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def get_optional_import(
    package_name: str,
    import_name: Optional[str] = None,
    warning_msg: Optional[str] = None,
):
    """
    Get optional import with fallback

    Args:
            package_name: Name of the package to import
            import_name: Name to use for import
            warning_msg: Custom warning message

    Returns:
            module or None: The imported module or None if not available
    """
    import_name = import_name or package_name
    warning_msg = (
        warning_msg or f"{package_name} not available. Some features will be limited."
    )

    try:
        return __import__(import_name)
    except ImportError:
        warnings.warn(warning_msg)
        return None


# Check all optional dependencies
HAS_SEABORN = check_optional_dependency("seaborn")
HAS_PLOTLY = check_optional_dependency("plotly")
HAS_SCIPY = check_optional_dependency("scipy")
HAS_SKLEARN = check_optional_dependency("sklearn")
HAS_STATSMODELS = check_optional_dependency("statsmodels")
HAS_HOLOVIEWS = check_optional_dependency("holoviews")


def check_dependencies():
    """Enhanced dependency checker"""
    print("Research Plotting Dependencies:")
    print("=" * 40)

    # Required dependencies
    required = [
        ("Matplotlib", True, "core plotting"),
        ("Polars", True, "data handling"),
        ("NumPy", True, "numerical operations"),
    ]

    for name, available, description in required:
        status = "✓" if available else "✗"
        print(f"- {name}: {status} (required - {description})")

    print("\nOptional Dependencies:")
    print("-" * 25)

    # Optional dependencies
    optional = [
        ("Seaborn", HAS_SEABORN, "enhanced research plots"),
        ("Plotly", HAS_PLOTLY, "interactive visualization"),
        ("SciPy", HAS_SCIPY, "advanced statistics & smoothing"),
        ("Scikit-learn", HAS_SKLEARN, "ML visualization support"),
        ("Statsmodels", HAS_STATSMODELS, "time series & statistical plots"),
        ("HoloViews", HAS_HOLOVIEWS, "exploratory visualization"),
    ]

    for name, available, description in optional:
        status = "✓" if available else "✗"
        print(f"- {name}: {status} (optional - {description})")

    # Installation commands
    missing_deps = []
    if not HAS_SEABORN:
        missing_deps.append("seaborn")
    if not HAS_PLOTLY:
        missing_deps.append("plotly kaleido")
    if not HAS_SCIPY:
        missing_deps.append("scipy")
    if not HAS_SKLEARN:
        missing_deps.append("scikit-learn")
    if not HAS_STATSMODELS:
        missing_deps.append("statsmodels")
    if not HAS_HOLOVIEWS:
        missing_deps.append("holoviews bokeh")

    if missing_deps:
        print(f"\nInstallation commands:")
        print("-" * 20)
        for dep in missing_deps:
            print(f"pip install {dep}")
    else:
        print("\n✓ All optional dependencies available!")
