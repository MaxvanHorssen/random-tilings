from ._RT import Aztec,Hexagon,config
__all__ = ['Aztec','Hexagon','config']

def _disable_jupyter_output_cache():
    try:
        from IPython import get_ipython

        ip = get_ipython()

        # Only active inside IPython/Jupyter
        if ip is None:
            return

        # Check whether this is a Jupyter kernel
        if "IPKernelApp" not in ip.config:
            return

        ip.cache_size = 0

        if hasattr(ip, "displayhook"):
            ip.displayhook.cache_size = 0
            ip.displayhook.flush()

    except Exception:
        # Never let cache handling break package import
        pass


_disable_jupyter_output_cache()

# Remove helper function from package namespace
del _disable_jupyter_output_cache

