from typing import Dict, Tuple, Callable, Sequence
import numpy as np
from mobile_env.core.movement import Movement
from mobile_env.core.entities import UserEquipment

class PositionGenerator(object):
    """Type of position generator"""

    @staticmethod
    def uniform_position(rng: np.random.Generator,
                         width_box: Tuple[float, float], height_box: Tuple[float, float]) -> Tuple[float, float]:
        """Generate a uniform position in a box

        Args:
            rng (np.random.Generator): random number generator
            width (Tuple[float]): minimum/maximum width
            height (Tuple[float]): minimum/maximum height

        Returns:
            Tuple[float, float]: random position in [width[0], width[1]]x[height[0], height[1]]
        """
        x = rng.uniform(width_box[0], width_box[1])
        y = rng.uniform(height_box[0], height_box[1])
        return (x,y)
    
    @staticmethod
    def radial_position(rng: np.random.Generator, center: Tuple[float, float], radius: float) -> Tuple[float, float]:
        """Generate a random position on a circle

        Args:
            rng (np.random.Generator): random number generator
            center (Tuple[float, float]): center of the circle (in x,y coordinates)
            radius (float): radius of the circle

        Returns:
            Tuple[float, float]: (x,y) position along the circle
        """
        theta = rng.uniform(-np.pi, np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        return (x,y)
    
    @staticmethod
    def linear_position(rng: np.random.Generator, center: Tuple[float, float],
                        angle: float, width_box: Tuple[float, float], height_box: Tuple[float, float]) -> Tuple[float, float]:
        """Generate a random position along a specified direction. The direction is given by
           the angle.

        Args:
            rng (np.random.Generator): random number generator
            center (Tuple[float, float]): initial position to consider
            angle (float): angle that defines the direction
            width_box (Tuple[float, float]): minimum/maximum width
            height_box (Tuple[float, float]): minimum/maximum height

        Returns:
            Tuple[float, float]: generated position (x,y)
        """
        cos_th = np.cos(angle)
        sin_th = np.sin(angle)
        maximum_radius_x = (width_box[1] * (cos_th >= 0.) + width_box[0] * (cos_th < 0.) - center[0]) / cos_th
        maximum_radius_y = (height_box[1] * (sin_th >= 0.) + height_box[0] * (sin_th < 0.) - center[1]) / sin_th
        
        maximum_radius = min(maximum_radius_x, maximum_radius_y)

        radius = rng.uniform(0, maximum_radius)
        x = center[0] + radius * cos_th
        y = center[1] + radius * sin_th
        return (x,y)


class RandomMovement(Movement):
    def __init__(self,
                 initial_position_generator: Sequence[Callable[[np.random.Generator], Tuple[int,int]]],
                 waypoint_position_generator: Sequence[Callable[[np.random.Generator], Tuple[int,int]] | None] | None,
                 **kwargs):
        """Random movement class for UEs

        Args:
            initial_position_generator (Sequence[Callable[[np.random.Generator], Tuple[int,int]]]): 
                                Sequence of functions to generate the random initial position for the UEs.
                                One for each UE.
            waypoint_position_generator (Sequence[Callable[[np.random.Generator], Tuple[int,int]] | None)]:
                                Sequence of functions to generate the random waypoints of the UEs. If None, this is
                                equivalent to a UE not moving. If the entire sequence is None, then all UEs are not moving.
        """
        super().__init__(**kwargs)

        # track waypoints and initial positions per UE
        self.waypoints: Dict[UserEquipment, Tuple[float, float]] = None
        self.initial: Dict[UserEquipment, Tuple[float, float]] = None
        self.initial_position_generator = initial_position_generator
        self.waypoint_position_generator = waypoint_position_generator

    def reset(self) -> None:
        super().reset()
        # NOTE: if RNG is not resetted after episode ends,
        # initial positions will differ between episodes
        self.waypoints = {}
        self.initial = {}

    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        """Move UE a step towards the random waypoint."""
        # generate random waypoint if UE has none so far
        if ue not in self.waypoints:
            # If the waypoint generator is none this is equivalent to no movement
            if self.waypoint_position_generator is None or self.waypoint_position_generator[ue.ue_id] is None:
                self.waypoints[ue] = (ue.x, ue.y)
            else:
                self.waypoints[ue] = self.waypoint_position_generator[ue.ue_id](self.rng)

        position = np.array([ue.x, ue.y])
        waypoint = np.array(self.waypoints[ue])

        # if already close enough to waypoint, move directly onto waypoint
        if np.linalg.norm(position - waypoint) <= ue.velocity:
            # remove waypoint from dict after it has been reached
            waypoint = self.waypoints.pop(ue)
            return waypoint

        # else move by self.velocity towards waypoint
        v = waypoint - position
        position = position + ue.velocity * v / np.linalg.norm(v)

        return tuple(position)

    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        """Return initial position of UE at the beginning of the episode."""
        if ue not in self.initial:
            self.initial[ue] = self.initial_position_generator[ue.ue_id](self.rng)

        x, y = self.initial[ue]
        return x, y
