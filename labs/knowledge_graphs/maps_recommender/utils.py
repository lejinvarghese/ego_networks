from datetime import datetime

from rich import print


class Logger:
    """A logger class with predefined color methods and custom color support."""

    color_palette = {
        "info": (229, 192, 123),  # warm yellow
        "success": (98, 209, 150),  # mint green
        "warning": (229, 165, 94),  # warm orange
        "error": (214, 97, 107),  # red
        "highlight": (97, 175, 239),  # blue
        "neutral": (198, 120, 221),  # purple
    }

    def __init__(self):
        for color_name, rgb in self.color_palette.items():
            setattr(self, color_name, self._create_color_method(rgb))

    def _get_timestamp(self):
        return datetime.now().strftime("[dim][grey30]%Y-%m-%d %H:%M:%S[/][/]")

    def _create_color_method(self, rgb):
        """Creates a method that prints in the specified RGB color."""

        def color_method(message: str):
            r, g, b = rgb
            color_name = next(
                name
                for name, value in self.color_palette.items()
                if value == rgb
            )
            timestamp = f"{self._get_timestamp()}"
            if color_name == "error":
                print(f"{timestamp} [rgb({r},{g},{b}) bold]{message}[/]")
            elif color_name == "warning":
                print(f"{timestamp} [rgb({r},{g},{b})]{message}[/]")
            else:
                print(f"{timestamp} [rgb({r},{g},{b})]{message}[/]")

        return color_method

    def print(self, message: str):
        timestamp = f"{self._get_timestamp()}"
        print(f"{timestamp} {message}")


logger = Logger()
