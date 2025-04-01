from pipeline.epde_translator.translation_structures import LLMPool
from pipeline.epde_translator.solution_translator import SolutionTranslator


class SolTrackTranslator(object):
    def __init__(self, record_track: dict, pruned_track: dict, dir_name: str):
        self.record_track = record_track
        self.pruned_track = pruned_track
        self.llm_pool = LLMPool()
        self.dir_name = dir_name
        self.eq_epde_str = None

    def translate(self):
        self.eq_epde_str = []
        for eq_key in self.pruned_track.keys():
            eq_str = SolutionTranslator(self.record_track[eq_key].rs_code,
                                        self.record_track[eq_key].params,
                                        self.llm_pool, self.dir_name).translate()
            if eq_str is not None:
                self.eq_epde_str.append(eq_str)
        assert len(self.eq_epde_str) != 0, 'The track for translation into epde code returned empty list'
        return self.eq_epde_str


if __name__ == '__main__':
    pop_track = {'du/dt = c[0] * du/dx + c[1] * t * du/dx + c[2] * t * x': (1.6, 460.5686610664196), 'du/dt = c[0] * du/dx + c[1] * u + c[2] * d^2u/dx^2': (1.45, 484.1114426561667), 'du/dt = c[0] * du/dx + c[1] * u * du/dx': (1.2, 438.94292729549943), 'du/dt = c[0] * u * du/dx + c[1] * du/dx + c[2] * d^2u/dx^2': (1.95, 37.14800565887713), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2': (1.45, 38.90635312678824), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2 + c[2] * du/dx * t': (2.15, 37.057907826954576), 'du/dt = c[0] * du/dx + c[1] * du/dt * d^2u/dx^2': (1.75, 542.9853705131861), 'du/dt = c[0] * du/dx + c[1] * t * du/dx': (1.2, 442.49077370655203)}

    print()