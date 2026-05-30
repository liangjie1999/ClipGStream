import viser
from nerfview import Viewer, RenderTabState
from typing import Literal
from typing import Callable, Tuple


# @dataclasses.dataclass
# class RenderTabState:
#     """Useful GUI handles exposed by the render tab."""

#     num_train_rays_per_sec: Optional[float] = None
#     num_view_rays_per_sec: float = 100000.0
#     preview_render: bool = False
#     preview_fov: float = 0.0
#     preview_time: float = 0.0
#     preview_aspect: float = 1.0
#     viewer_res: int = 2048
#     viewer_width: int = 1280
#     viewer_height: int = 960
#     render_width: int = 1280
#     render_height: int = 960

class BetaRenderTabState(RenderTabState):
    # non-controlable parameters
    total_count_number: int = 0
    rendered_count_number: int = 0

    # controlable parameters
    near_plane: float = 1e-3
    far_plane: float = 1e3
    radius_clip: float = 0.0
    b_range: Tuple[float, float] = (-5.0, 5.0)
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal["RGB", "Alpha", "Diffuse", "Specular", "Depth", "Normal"] = (
        "RGB")
    time_stamp: int = 0


# 这个可视化工具 非常值得研究 要是能让在服务器上 适配各种光栅化方式就好了，感觉应该是可以的
class BetaViewer(Viewer):
    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        mode: Literal["rendering", "training"] = "rendering",
        share_url: bool = False,
    ):
        super().__init__(server, render_fn, mode=mode)  # 这个render_fn是传给第三方对象的（继承自Viewer）得看看Viewer这个类
        server.gui.set_panel_label("Beta Splatting Viewer") # 设置标题
        if share_url:
            server.request_share_url()  # ？这啥意思 能看远程得难不成？
    
    # 这个看着像是一些生命周期函数
    def _init_rendering_tab(self):
        self.render_tab_state = BetaRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        with self._rendering_folder:
            with self.server.gui.add_folder("Geometry Complexity Control"):
                self.gui_multi_slider = self.server.gui.add_multi_slider(
                    "b Range",
                    min=-5,
                    max=5,
                    step=0.01,
                    initial_value=self.render_tab_state.b_range,
                )

                @self.gui_multi_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.b_range = self.gui_multi_slider.value
                    self.rerender(_)

            with self.server.gui.add_folder("Render Mode"):
                self.render_mode_dropdown = self.server.gui.add_dropdown(
                    "Mode",
                    ["RGB", "Alpha", "Diffuse", "Specular", "Depth", "Normal"],
                    initial_value=self.render_tab_state.render_mode,
                )

                @self.render_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.render_mode = self.render_mode_dropdown.value
                    self.rerender(_)

                self.total_count_number = self.server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_count_number,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                self.rendered_count_number = self.server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_count_number,
                    disabled=True,
                    hint="Number of splats rendered.",
                )
                self.radius_clip_slider = self.server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                self.time_stamp_slider = self.server.gui.add_number(
                    "Time Stamp",
                    initial_value=self.render_tab_state.time_stamp,
                    min=0,
                    max=200,
                    step=1,
                    hint="current time stamp",
                )

                @self.radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = self.radius_clip_slider.value
                    self.rerender(_)

                @self.time_stamp_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.time_stamp = self.time_stamp_slider.value
                    self.rerender(_)

                self.near_far_plane_vec2 = self.server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @self.near_far_plane_vec2.on_update
                def _(_) -> None:
                    (
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ) = self.near_far_plane_vec2.value
                    self.rerender(_)

                self.backgrounds_slider = self.server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @self.backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = self.backgrounds_slider.value
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "b_range": self.gui_multi_slider,
                "total_count_number": self.total_count_number,
                "rendered_count_number": self.rendered_count_number,
                "near_far_plane_vec2": self.near_far_plane_vec2,
                "radius_clip_slider": self.radius_clip_slider,
                "rener_mode_dropdown": self.render_mode_dropdown,
                "backgrounds_slider": self.backgrounds_slider,
                "time_stamp_slider": self.time_stamp_slider
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles["total_count_number"].value = (
            self.render_tab_state.total_count_number
        )
        self._rendering_tab_handles["rendered_count_number"].value = (
            self.render_tab_state.rendered_count_number
        )
