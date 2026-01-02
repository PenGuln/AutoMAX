"""
GUI main entry point for AutoAUC.
"""

import logging
import flet as ft
from .gui import AutoAUCApp
from .monitor import MonitorPage

# Configure logging BEFORE any session manager is created
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auto_auc.log')
    ]
)


def main():
    """Main GUI entry point."""
    def main_page(page: ft.Page):
        page.title = "AutoMAX"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.window.width = 1200
        page.window.height = 800
        page.window.resizable = True
        # page.scroll = ft.ScrollMode.ADAPTIVE
        
        # Create the main app
        app = AutoAUCApp()
        app.initialize()
        monitor_page = MonitorPage()
        
        # Set up routing
        def route_change(route):
            page.views.clear()
            
            if page.route == "/":
                page.views.append(
                    ft.View(
                        "/",
                        [app],
                        scroll=ft.ScrollMode.ADAPTIVE
                    )
                )
            if page.route == "/monitor":
                page.views.append(
                    ft.View(
                        "/monitor",
                        [monitor_page],
                        scroll=ft.ScrollMode.ADAPTIVE
                    )
                )
            
            page.update()
            
            if page.route == "/monitor":
                monitor_page.initialize()
        
        page.on_route_change = route_change
        page.go(page.route)
    
    # Run the Flet app
    ft.app(target=main_page)


if __name__ == "__main__":
    main()
