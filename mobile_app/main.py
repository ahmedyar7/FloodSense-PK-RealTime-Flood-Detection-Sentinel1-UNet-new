import asyncio
import base64
import json
import os

import flet as ft
from dotenv import load_dotenv

import alert_engine as engine
import database as db

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def main(page: ft.Page):
    page.title = "FloodSense-PK"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.window.width = 400
    page.window.height = 800
    page.bgcolor = ft.Colors.BLACK

    db.init_db()

    def load_districts(path: str) -> list:
        if not os.path.exists(path):
            return ["Swat", "Nowshera", "Charsadda", "Peshawar", "Sukkur", "Larkana"]
        raw = open(path, "rb").read()
        for encoding in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le"):
            try:
                return json.loads(raw.decode(encoding))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        raise ValueError(f"Could not read districts file: {path}")

    districts = load_districts(os.path.join(APP_DIR, "districts.json"))

    state = {
        "user": None,
        "district": None,
        "nav_index": 0,
        "cache": {},
        "home_fetch_token": 0,
    }

    def navigate(route: str):
        asyncio.create_task(page.push_route(route))

    def safe_update():
        try:
            page.update()
        except RuntimeError:
            pass

    def on_nav_change(e):
        state["nav_index"] = e.control.selected_index
        if state["nav_index"] == 0:
            navigate("/home")
        elif state["nav_index"] == 1:
            navigate("/map")
        elif state["nav_index"] == 2:
            navigate("/profile")

    def get_nav_bar():
        return ft.NavigationBar(
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.HOME_OUTLINED, selected_icon=ft.Icons.HOME, label="Home"),
                ft.NavigationBarDestination(icon=ft.Icons.MAP_OUTLINED, selected_icon=ft.Icons.MAP, label="Analysis"),
                ft.NavigationBarDestination(icon=ft.Icons.PERSON_OUTLINED, selected_icon=ft.Icons.PERSON, label="Profile"),
            ],
            selected_index=state["nav_index"],
            on_change=on_nav_change,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        )

    def login_view():
        username_field = ft.TextField(
            label="Username",
            prefix_icon=ft.Icons.PERSON,
            border_radius=15,
            filled=True,
        )
        password_field = ft.TextField(
            label="Password",
            password=True,
            can_reveal_password=True,
            prefix_icon=ft.Icons.LOCK,
            border_radius=15,
            filled=True,
        )
        error_text = ft.Text(color=ft.Colors.RED_400)

        def handle_login(e):
            username = (username_field.value or "").strip()
            password = password_field.value or ""

            if not username or not password:
                error_text.value = "Please enter username and password"
                safe_update()
                return

            print(f"Attempting login for: {username}")
            user = db.login_user(username, password)
            if user:
                print(f"Login successful for {user[1]}. Navigating to /home")
                state["user"] = user
                state["district"] = user[2]
                state["cache"] = {}
                state["nav_index"] = 0
                navigate("/home")
            else:
                print("Login failed: Invalid credentials")
                error_text.value = "Invalid username or password"
                safe_update()

        return ft.View(
            route="/login",
            padding=30,
            vertical_alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Icon(ft.Icons.WATER_DROP, size=100, color=ft.Colors.BLUE_400),
                            ft.Text("FloodSense-PK", size=32, weight=ft.FontWeight.BOLD),
                            ft.Text("National Intelligence Portal", size=16, color=ft.Colors.BLUE_200),
                            ft.Divider(height=40, color=ft.Colors.TRANSPARENT),
                            username_field,
                            password_field,
                            error_text,
                            ft.Button(
                                "Login",
                                on_click=handle_login,
                                width=float("inf"),
                                height=50,
                                style=ft.ButtonStyle(
                                    bgcolor=ft.Colors.BLUE_700,
                                    color=ft.Colors.WHITE,
                                    shape=ft.RoundedRectangleBorder(radius=15),
                                ),
                            ),
                            ft.TextButton("Create an account", on_click=lambda _: navigate("/register")),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                )
            ],
        )

    def register_view():
        username_field = ft.TextField(label="Username", prefix_icon=ft.Icons.PERSON, border_radius=15, filled=True)
        password_field = ft.TextField(
            label="Password", password=True, can_reveal_password=True, prefix_icon=ft.Icons.LOCK, border_radius=15, filled=True
        )
        district_dropdown = ft.Dropdown(
            label="Select District",
            options=[ft.dropdown.Option(d) for d in districts],
            leading_icon=ft.Icons.LOCATION_ON,
            border_radius=15,
            filled=True,
        )
        error_text = ft.Text(color=ft.Colors.RED_400)

        def handle_register(e):
            username = (username_field.value or "").strip()
            password = password_field.value or ""
            district = district_dropdown.value

            if not username or not password or not district:
                error_text.value = "All fields are required"
                safe_update()
                return

            success = db.register_user(username, password, district)
            if success:
                navigate("/login")
            else:
                error_text.value = "Username already exists"
                safe_update()

        return ft.View(
            route="/register",
            padding=30,
            vertical_alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Text("Join the Network", size=28, weight=ft.FontWeight.BOLD),
                ft.Text("Monitor flood risks in your area", size=16, color=ft.Colors.GREY_400),
                ft.Divider(height=30, color=ft.Colors.TRANSPARENT),
                username_field,
                password_field,
                district_dropdown,
                error_text,
                ft.Button(
                    "Register",
                    on_click=handle_register,
                    width=float("inf"),
                    height=50,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE, shape=ft.RoundedRectangleBorder(radius=15)),
                ),
                ft.TextButton("Back to Login", on_click=lambda _: navigate("/login")),
            ],
        )

    def home_view():
        district_name = state["district"]
        content_area = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=20)
        loading_indicator = ft.ProgressBar(width=400, color=ft.Colors.BLUE)

        def render_alerts(data):
            content_area.controls.clear()

            if "error" in data:
                content_area.controls.append(ft.Text(f"Error: {data['error']}", color=ft.Colors.RED_400))
                content_area.controls.append(
                    ft.Button("Retry", on_click=lambda _: refresh_alerts(force=True), icon=ft.Icons.REFRESH)
                )
                return

            content_area.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(f"Hi, {state['user'][1]}", size=24, weight=ft.FontWeight.BOLD),
                            ft.Text(f"Monitoring {district_name}", color=ft.Colors.BLUE_200),
                        ]
                    ),
                    margin=ft.Margin.only(bottom=10),
                )
            )

            risk_color = ft.Colors.GREEN_400
            if data["risk_score"] > 7:
                risk_color = ft.Colors.RED_400
            elif data["risk_score"] > 4:
                risk_color = ft.Colors.ORANGE_400

            content_area.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Icon(ft.Icons.WARNING_ROUNDED, color=risk_color, size=30),
                                    ft.Text(f"Risk Level: {data['risk_score']}/10", size=20, weight=ft.FontWeight.BOLD),
                                ]
                            ),
                            ft.Text("Based on current satellite & river metrics", size=12, color=ft.Colors.GREY_400),
                            ft.Divider(height=20, color=ft.Colors.WHITE10),
                            ft.Row(
                                [
                                    ft.Column(
                                        [
                                            ft.Text("CURRENT", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_500),
                                            ft.Text(f"{data['pct_current']}%", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_200),
                                        ],
                                        expand=True,
                                    ),
                                    ft.Column(
                                        [
                                            ft.Text("HISTORIC (2010)", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_500),
                                            ft.Text(f"{data['pct_2010']}%", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_400),
                                        ],
                                        expand=True,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    padding=20,
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=20,
                )
            )

            if data.get("has_river"):
                content_area.controls.append(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("Hydraulic Status", size=16, weight=ft.FontWeight.BOLD),
                                ft.ListTile(
                                    leading=ft.Icon(ft.Icons.WATER_DROP, color=ft.Colors.BLUE_400),
                                    title=ft.Text(data["station"]),
                                    subtitle=ft.Text(f"River Condition: {data['status']}"),
                                    content_padding=0,
                                ),
                                ft.Row(
                                    [
                                        ft.Text(f"Inflow: {data['inflow']}", size=12),
                                        ft.VerticalDivider(),
                                        ft.Text(f"Outflow: {data['outflow']}", size=12),
                                    ],
                                    alignment=ft.MainAxisAlignment.START,
                                ),
                            ]
                        ),
                        padding=20,
                        bgcolor=ft.Colors.BLUE_900_DARK,
                        border_radius=20,
                        opacity=0.8,
                    )
                )

            content_area.controls.append(
                ft.Text(
                    f"Last updated: {data['timestamp']}",
                    size=11,
                    italic=True,
                    color=ft.Colors.GREY_600,
                    text_align=ft.TextAlign.CENTER,
                )
            )
            content_area.controls.append(
                ft.Button("Refresh", on_click=lambda _: refresh_alerts(force=True), icon=ft.Icons.REFRESH, width=float("inf"))
            )

        def refresh_alerts(e=None, force=False):
            if state.get("cache") and not force and "error" not in state.get("cache", {}):
                render_alerts(state["cache"])
                safe_update()
                return

            state["home_fetch_token"] += 1
            fetch_token = state["home_fetch_token"]

            content_area.controls.clear()
            content_area.controls.append(loading_indicator)
            safe_update()

            district = state["district"]

            def fetch_and_render():
                data = engine.get_district_alert(district)

                if fetch_token != state["home_fetch_token"] or page.route != "/home":
                    return

                state["cache"] = data
                render_alerts(data)
                safe_update()

            page.run_thread(fetch_and_render)

        if state.get("cache"):
            render_alerts(state["cache"])
        else:
            refresh_alerts()

        return ft.View(
            route="/home",
            padding=ft.Padding.only(left=20, right=20, top=20),
            navigation_bar=get_nav_bar(),
            controls=[content_area],
        )

    def map_view():
        data = state.get("cache")
        content = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=20)

        if not data or "map_image" not in data:
            content.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Icon(ft.Icons.MAP_OUTLINED, size=50, color=ft.Colors.GREY_700),
                            ft.Text("No analysis data available.\nPlease refresh from Home.", text_align=ft.TextAlign.CENTER),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    alignment=ft.Alignment.CENTER,
                    expand=True,
                )
            )
        else:
            content.controls.append(ft.Text("Satellite Analysis", size=24, weight=ft.FontWeight.BOLD))
            content.controls.append(
                ft.Container(
                    content=ft.Image(
                        src=f"data:image/png;base64,{data['map_image']}",
                        fit=ft.BoxFit.CONTAIN,
                        border_radius=20,
                    ),
                    bgcolor=ft.Colors.BLACK,
                    border_radius=20,
                )
            )
            content.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Map Legend", size=14, weight=ft.FontWeight.BOLD),
                            ft.Row(
                                [
                                    ft.Container(width=12, height=12, bgcolor=ft.Colors.BLUE, border_radius=3),
                                    ft.Text("Detected Flood Water", size=12),
                                ]
                            ),
                            ft.Row(
                                [
                                    ft.Container(width=12, height=12, bgcolor=ft.Colors.GREY_700, border_radius=3),
                                    ft.Text("Terrain (Radar Imagery)", size=12),
                                ]
                            ),
                        ],
                        spacing=5,
                    ),
                    padding=15,
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=15,
                )
            )

        return ft.View(
            route="/map",
            padding=20,
            navigation_bar=get_nav_bar(),
            controls=[content],
        )

    def profile_view():
        def logout(e):
            state["user"] = None
            state["district"] = None
            state["cache"] = {}
            state["nav_index"] = 0
            state["home_fetch_token"] += 1
            navigate("/login")

        return ft.View(
            route="/profile",
            padding=20,
            navigation_bar=get_nav_bar(),
            controls=[
                ft.Text("My Profile", size=24, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.ListTile(
                                leading=ft.CircleAvatar(content=ft.Text(state["user"][1][0].upper())),
                                title=ft.Text(state["user"][1], size=18, weight=ft.FontWeight.BOLD),
                                subtitle=ft.Text("Authorized User"),
                            ),
                            ft.Divider(),
                            ft.ListTile(
                                leading=ft.Icon(ft.Icons.LOCATION_ON),
                                title=ft.Text("Target District"),
                                subtitle=ft.Text(state["district"]),
                            ),
                        ]
                    ),
                    padding=10,
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=20,
                ),
                ft.Divider(height=40, color=ft.Colors.TRANSPARENT),
                ft.Button(
                    "Logout",
                    icon=ft.Icons.LOGOUT,
                    on_click=logout,
                    width=float("inf"),
                    style=ft.ButtonStyle(color=ft.Colors.RED_200),
                ),
            ],
        )

    def route_change(e):
        print(f"Route changed to: {page.route}")

        if page.route in ("/home", "/map", "/profile") and not state["user"]:
            page.views.clear()
            page.views.append(login_view())
            safe_update()
            return

        page.views.clear()
        if page.route == "/login":
            page.views.append(login_view())
        elif page.route == "/register":
            page.views.append(register_view())
        elif page.route == "/home":
            state["nav_index"] = 0
            page.views.append(home_view())
        elif page.route == "/map":
            state["nav_index"] = 1
            page.views.append(map_view())
        elif page.route == "/profile":
            state["nav_index"] = 2
            page.views.append(profile_view())
        else:
            page.views.append(login_view())
        safe_update()

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.route = top_view.route
        safe_update()

    page.on_route_change = route_change
    page.on_view_pop = view_pop

    print("Starting app...")
    page.route = "/login"
    route_change(None)


if __name__ == "__main__":
    ft.run(main)
