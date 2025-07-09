"""Info module."""

# Authors: Hamza Cherkaoui

from distutils.version import LooseVersion


__version__ = "0.1.dev"

PACKAGE_NAME = 'agoralearn'
GITHUB_REPO_URL = "https://github.com/hcherkaoui/agoralearn"
INSTALL_MSG = f"See {GITHUB_REPO_URL} for installation" f" information."
REQUIRED_MODULE_METADATA = (
    (
        "numba",
        {
            "min_version": "0.61.2",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "numpy",
        {
            "min_version": "2.2.6",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "pandas",
        {
            "min_version": "2.3.1",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "torch",
        {
            "min_version": "2.7.1",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "torchvision",
        {
            "min_version": "0.22.1",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "transformers",
        {
            "min_version": "4.53.1",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "joblib",
        {
            "min_version": "0.16.0",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "scikit-learn",
        {
            "min_version": "1.7.0",
            "import_name": "sklearn",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "seaborn",
        {
            "min_version": "0.13.2",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
    (
        "matplotlib",
        {
            "min_version": "3.10.0",
            "required_at_installation": True,
            "install_info": INSTALL_MSG,
        },
    ),
)


def _import_module_with_version_check(module_name, minimum_version, install_info=None):
    """Private helper, check that module is installed with a recent enough
    version.

    Parameters
    ----------
    module_name : str, module name
    minimum_version : str, minimum version required
    install_info : str or None, (default=None), message to install it if
        installation failed

    Return
    ------
    module : Python module, the imported module
    """
    try:
        module = __import__(module_name)

    except ImportError as exc:
        msg = f"Please install it properly to use {PACKAGE_NAME}."
        user_friendly_info = f"Module '{module_name}' could not be found. " f"{install_info or msg}"
        exc.args += (user_friendly_info,)
        raise

    module_version = getattr(module, "__version__", "0.0.0")

    version_too_old = not LooseVersion(module_version) >= LooseVersion(minimum_version)

    if version_too_old:
        raise ImportError(f"A {module_name} version of at least {minimum_version} "
                          f"is required to use {PACKAGE_NAME}. {module_version} was "
                          f"found. Please upgrade {module_name}")

    return module


def _check_module_dependencies(is_installing=False):
    """Throw an exception if Bandpy dependencies are not installed.

    Parameters
    ----------
    is_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.
    Throws
    ------
    ImportError : if a dependencie is not installed.
    """

    for module_name, module_metadata in REQUIRED_MODULE_METADATA:
        if not (
            is_installing and not module_metadata["required_at_installation"]
        ):
            if "import_name" in module_metadata.keys():
                module_name = module_metadata["import_name"]
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata["min_version"],
                install_info=module_metadata.get("install_info"),
            )
