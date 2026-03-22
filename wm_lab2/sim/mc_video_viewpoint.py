from __future__ import annotations

"""
Malmo <VideoProducer> supports viewpoint (see MissionHandlers.xsd):
  0 = first-person, 1 = third-person behind, 2 = third-person facing.

MineDojo's POVObservation omits this attribute (defaults to 0). We monkey-patch
handlers.POVObservation before minedojo.make() so mission XML includes the
requested viewpoint. Minecraft applies it in VideoProducerImplementation.prepare().
"""


def install_pov_viewpoint(*, viewpoint: int) -> None:
    vp = int(viewpoint)
    if vp not in (0, 1, 2):
        raise ValueError(f"mc_video_viewpoint must be 0, 1, or 2 (Malmo VideoProducer); got {vp}")

    import minedojo.sim.handlers as handlers  # type: ignore
    from minedojo.sim.handlers.agent.observations.pov import POVObservation as _Base  # type: ignore

    class POVObservationWithViewpoint(_Base):
        def __init__(self, video_resolution, include_depth=False):
            super().__init__(video_resolution, include_depth)
            self.viewpoint = vp

        def xml_template(self) -> str:
            return str(
                """
            <VideoProducer
                want_depth="{{ include_depth | string | lower }}"
                viewpoint="{{ viewpoint }}">
                <Width>{{ video_width }} </Width>
                <Height>{{ video_height }}</Height>
            </VideoProducer>"""
            )

    handlers.POVObservation = POVObservationWithViewpoint
