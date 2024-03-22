from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.core.util import deep_dict_merge


class SingleBSMComEnv(MComCore):
    def __init__(self, num_ues=5, config={}, render_mode: str | None=None):
        """Single Base Station environment with multiple UEs

        Args:
            num_ues (int, optional): number of UEs. Defaults to 5.
            config (dict, optional): configuration file. Defaults to {}.
            render_mode (str, optional): type of render. Defaults to None.
        """
        assert num_ues > 0 and isinstance(num_ues, int), 'num_ues needs to be a positive integer'

        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        station_pos = (config['width'] // 2, config['height'] // 2)
        stations = [BaseStation(0, station_pos, **config["bs"])]
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)]

        super().__init__(stations, ues, config, render_mode)

    # overwrite the default config
    @classmethod
    def default_config(cls):
        config = super().default_config()
        return config