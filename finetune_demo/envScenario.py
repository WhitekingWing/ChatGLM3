from typing import List, Tuple, Optional, Union, Dict
from datetime import datetime
import math
import os

from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.road.lane import (
    StraightLane, CircularLane, SineLane, PolyLane, PolyLaneFixedWidth
)
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import numpy as np


#from envPlotter import ScePlotter


ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle',
    4: 'Deceleration - decelerate the vehicle'
}


class EnvScenario:
    def __init__(
            self, env: AbstractEnv, envType: str,
            seed: int, database: str = None
    ) -> None:
        self.env = env
        self.envType = envType

        self.ego: MDPVehicle = env.vehicle
        # 下面的四个变量用来判断车辆是否在 ego 的危险视距内
        self.theta1 = math.atan(3/17.5)
        self.theta2 = math.atan(2/2.5)
        self.radius1 = np.linalg.norm([3, 17.5])
        self.radius2 = np.linalg.norm([2, 2.5])

        self.road: Road = env.road
        self.network: RoadNetwork = self.road.network

        #self.plotter = ScePlotter()
        if database:
            self.database = database
        else:
            self.database = datetime.strftime(
                datetime.now(), '%Y-%m-%d_%H-%M-%S'
            ) + '.db'

        if os.path.exists(self.database):
            os.remove(self.database)



    def getSurrendVehicles(self, vehicles_count: int) -> List[IDMVehicle]:
        return self.road.close_vehicles_to(
            self.ego, self.env.PERCEPTION_DISTANCE,
            count=vehicles_count-1, see_behind=True,
            sort='sorted'
        )

    def plotSce(self, fileName: str) -> None:
        SVs = self.getSurrendVehicles(10)
        self.plotter.plotSce(self.network, SVs, self.ego, fileName)

    def getUnitVector(self, radian: float) -> Tuple[float]:
        return (
            math.cos(radian), math.sin(radian)
        )

    def isInJunction(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:
        if self.envType == 'intersection-v1':
            x, y = vehicle.position
            # 这里交叉口的范围是 -12~12, 这里是为了保证车辆可以检测到交叉口内部的信息
            # 这个时候车辆需要提前减速
            if -20 <= x <= 20 and -20 <= y <= 20:
                return True
            else:
                return False
        else:
            return False

    def getLanePosition(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:
        currentLaneIdx = vehicle.lane_index
        currentLane = self.network.get_lane(currentLaneIdx)
        if not isinstance(currentLane, StraightLane):
            raise ValueError(
                "车辆在交叉路口，无法获取车道位置。"
            )
        else:
            currentLane = self.network.get_lane(vehicle.lane_index)
            return np.linalg.norm(vehicle.position - currentLane.start)

    def availableActionsDescription(self) -> str:
        avaliableActionDescription = 'Your available actions are: '
        availableActions = self.env.get_available_actions()
        for action in availableActions:
            avaliableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(
                action) + ''
        # if 1 in availableActions:
        #     avaliableActionDescription += 'You should check IDLE action as FIRST priority. '
        # if 0 in availableActions or 2 in availableActions:
        #     avaliableActionDescription += 'For change lane action, CAREFULLY CHECK the safety of vehicles on target lane. '
        # if 3 in availableActions:
        #     avaliableActionDescription += 'Consider acceleration action carefully. '
        # if 4 in availableActions:
        #     avaliableActionDescription += 'The deceleration action is LAST priority. '
        # avaliableActionDescription += ''
        return avaliableActionDescription

    def processNormalLane(self, lidx: LaneIndex) -> str:
        sideLanes = self.network.all_side_lanes(lidx)
        numLanes = len(sideLanes)
        if numLanes == 1:
            description = "您正在一条只有一条车道的路上行驶，您无法变道。 "
        else:
            egoLaneRank = lidx[2]
            if egoLaneRank == 0:
                description = f"您正在一条有{numLanes}条车道的路上行驶，目前您在最左侧车道上。"
            elif egoLaneRank == numLanes - 1:
                description = f"您正在一条有{numLanes}条车道的路上行驶，目前您在最右侧车道上。"
            else:
                laneRankDict = {
                    1: '第二条',
                    2: '第三条',
                    3: '第四条'
                }
                description = f"您正在一条有{numLanes}条车道的路上行驶，目前您在从右数{laneRankDict[numLanes - egoLaneRank - 1]}车道上。"

        description += f"您的当前位置是({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})，速度为{self.ego.speed:.2f}米/秒，加速度为{self.ego.action['acceleration']:.2f}米/秒^2，车道位置为{self.getLanePosition(self.ego):.2f}米。"
        return description

    def getSVRelativeState(self, sv: IDMVehicle) -> str:
        # CAUTION: 这里有一个问题，pygame 的 y 轴是上下颠倒的，向下是 y 轴的正方向。
        #       因此，在 highway-v0 上，车辆向左换道实际上是向右运动。因此判断车辆相
        #       对自车的位置，不能用向量来算，直接根据车辆在哪条车道上来判断是比较合适
        #       的，向量只能用来判断车辆在 ego 的前方还是后方
        relativePosition = sv.position - self.ego.position
        egoUnitVector = self.getUnitVector(self.ego.heading)
        cosineValue = sum(
            [x*y for x, y in zip(relativePosition, egoUnitVector)]
        )
        if cosineValue >= 0:
            return '位于您前方'
        else:
            return '位于您后方'

    def getVehDis(self, veh: IDMVehicle):
        posA = self.ego.position
        posB = veh.position
        distance = np.linalg.norm(posA - posB)
        return distance

    def getClosestSV(self, SVs: List[IDMVehicle]):
        if SVs:
            closestIdex = -1
            closestDis = 99999999
            for i, sv in enumerate(SVs):
                dis = self.getVehDis(sv)
                if dis < closestDis:
                    closestDis = dis
                    closestIdex = i
            return SVs[closestIdex]
        else:
            return None

    def processSingleLaneSVs(self, SingleLaneSVs: List[IDMVehicle]):
        # 返回当前车道上，前方最近的车辆和后方最近的车辆，如果没有，则为 None
        if SingleLaneSVs:
            aheadSVs = []
            behindSVs = []
            for sv in SingleLaneSVs:
                RSStr = self.getSVRelativeState(sv)
                if RSStr == '位于您前方':
                    aheadSVs.append(sv)
                else:
                    behindSVs.append(sv)
            aheadClosestOne = self.getClosestSV(aheadSVs)
            behindClosestOne = self.getClosestSV(behindSVs)
            return aheadClosestOne, behindClosestOne
        else:
            return None, None

    def processSVsNormalLane(
            self, SVs: List[IDMVehicle], currentLaneIndex: LaneIndex
    ):
        # 目前 description 中的车辆有些太多了，需要处理一下，只保留最靠近 ego 的几辆车
        classifiedSVs: Dict[str, List[IDMVehicle]] = {
            'current lane': [],
            'left lane': [],
            'right lane': [],
            'target lane': []
        }
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        for sv in SVs:
            lidx = sv.lane_index
            if lidx in sideLanes:
                if lidx == currentLaneIndex:
                    classifiedSVs['current lane'].append(sv)
                else:
                    laneRelative = lidx[2] - currentLaneIndex[2]
                    if laneRelative == 1:
                        classifiedSVs['right lane'].append(sv)
                    elif laneRelative == -1:
                        classifiedSVs['left lane'].append(sv)
                    else:
                        continue
            elif lidx == nextLane:
                classifiedSVs['target lane'].append(sv)
            else:
                continue

        validVehicles: List[IDMVehicle] = []
        existVehicles: Dict[str, bool] = {}
        for k, v in classifiedSVs.items():
            if v:
                existVehicles[k] = True
            else:
                existVehicles[k] = False
            ahead, behind = self.processSingleLaneSVs(v)
            if ahead:
                validVehicles.append(ahead)
            if behind:
                validVehicles.append(behind)

        return validVehicles, existVehicles

    def describeSVNormalLane(self, currentLaneIndex: LaneIndex) -> str:
        # 当 ego 在 StraightLane 上时，车道信息是重要的，需要处理车道信息
        # 首先判断车辆是不是和车辆在同一条 road 上
        #   如果在同一条 road 上，则判断在哪条 lane 上
        #   如果不在同一条 road 上，则判断是否在 next_lane 上
        #      如果不在 nextLane 上，则直接不考虑这辆车的信息
        #      如果在 nextLane 上，则统计这辆车关于 ego 的相对运动状态
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        surroundVehicles = self.getSurrendVehicles(10)
        validVehicles, existVehicles = self.processSVsNormalLane(
            surroundVehicles, currentLaneIndex
        )
        if not surroundVehicles:
            SVDescription = "周围没有其他车辆行驶，所以您可以完全根据自己的想法驾驶。"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles:
                lidx = sv.lane_index
                if lidx in sideLanes:
                    # 车辆和 ego 在同一条 road 上行驶
                    if lidx == currentLaneIndex:
                        # 车辆和 ego 在同一条 lane 上行驶
                        if sv in validVehicles:
                            SVDescription += f"车辆{id(sv) % 1000}与您在同一车道，{self.getSVRelativeState(sv)}。"
                        else:
                            continue
                    else:
                        laneRelative = lidx[2] - currentLaneIndex[2]
                        if laneRelative == 1:
                            # laneRelative = 1 表示车辆在 ego 的右侧车道上行驶
                            if sv in validVehicles:
                                SVDescription += f"车辆{id(sv) % 1000}在您右侧的车道行驶，{self.getSVRelativeState(sv)}。"
                            else:
                                continue
                        elif laneRelative == -1:
                            # laneRelative = -1 表示车辆在 ego 的左侧车道上行驶
                            if sv in validVehicles:
                                SVDescription += f"车辆{id(sv) % 1000}在您左侧的车道行驶，{self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        else:
                            # laneRelative 是其他的值表示在更远的车道上，不需要考虑
                            continue
                elif lidx == nextLane:
                    # 车辆在 ego 的 nextLane 上行驶
                    if sv in validVehicles:
                        SVDescription += f"车辆{id(sv) % 1000}与您在同一车道，{self.getSVRelativeState(sv)}. "
                    else:
                        continue
                else:
                    continue
                if self.envType == 'intersection-v1':
                    SVDescription += f"它的位置是({sv.position[0]:.2f}, {sv.position[1]:.2f})，速度为{sv.speed:.2f}米/秒，加速度为{sv.action['acceleration']:.2f} 米/秒^2。"
                else:
                    SVDescription += f"它的位置是({sv.position[0]:.2f}, {sv.position[1]:.2f})，速度为{sv.speed:.2f}米/秒，加速度为{sv.action['acceleration']:.2f} 米/秒^2，车道位置为{self.getLanePosition(sv):.2f}米。"
            if SVDescription:
                descriptionPrefix = "周围还有其他车辆在行驶，以下是它们的基本信息："
                return descriptionPrefix + SVDescription
            else:
                SVDescription = '周围没有其他车辆行驶，所以您可以完全根据自己的想法驾驶。'
                return SVDescription

    def isInDangerousArea(self, sv: IDMVehicle) -> bool:
        relativeVector = sv.position - self.ego.position
        distance = np.linalg.norm(relativeVector)
        egoUnitVector = self.getUnitVector(self.ego.heading)
        relativeUnitVector = relativeVector / distance
        alpha = np.arccos(
            np.clip(np.dot(egoUnitVector, relativeUnitVector), -1, 1)
        )
        if alpha <= self.theta1:
            if distance <= self.radius1:
                return True
            else:
                return False
        elif self.theta1 < alpha <= self.theta2:
            if distance <= self.radius2:
                return True
            else:
                return False
        else:
            return False
    def getCollisionPoint(self, other_vehicle):
        # 获取ego车辆和其他车辆的位置和速度
        ego_position = self.ego.position
        ego_speed = self.ego.speed
        ego_heading = self.ego.heading  # 这应该是车辆的朝向向量
        other_position = other_vehicle.position
        other_speed = other_vehicle.speed
        other_heading = other_vehicle.heading  # 这应该是其他车辆的朝向向量
        # 计算两车之间的相对速度
        relative_speed = other_speed - ego_speed
        # 如果两车速度相同或背道而驰，它们不会碰撞
        if relative_speed == 0:
            return None
        # 计算两车之间的相对位置
        relative_position = other_position - ego_position
        # 计算两车朝向之间的角度差
        angle_difference = np.arccos(np.dot(ego_heading, other_heading))
        # 在交叉口中，车辆的轨迹可能更复杂
        # 我们需要考虑车辆的转向和可能的避让行为
        # 这里我们使用一个简化的模型来估计车辆的轨迹
        # 您可能需要根据您的具体应用场景和车辆运动模型来调整这个方法
        if self.envType == 'intersection-v1':
            # 假设交叉口的范围是 -12~12, 这里是为了保证车辆可以检测到交叉口内部的信息
            # 这个时候车辆需要提前减速
            x, y = other_position
            if -20 <= x <= 20 and -20 <= y <= 20:
                # 车辆在交叉口中
                # 为了简化，我们假设车辆在交叉口中时会立即停止
                # 因此，碰撞点就是当前位置
                print(ego_heading,other_heading)
                return other_position
    def describeSVJunctionLane(self, currentLaneIndex: LaneIndex) -> str:
        # 当 ego 在交叉口内部时，车道的信息不再重要，只需要判断车辆和 ego 的相对位置
        # 但是需要判断交叉口内部所有车道关于 ego 的位置
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        surroundVehicles = self.getSurrendVehicles(6)
        if not surroundVehicles:
            SVDescription = "周围没有其他车辆行驶，所以您可以完全根据自己的想法驾驶。"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles:
                lidx = sv.lane_index
                if self.isInJunction(sv):
                    collisionPoint = self.getCollisionPoint(sv)
                    if collisionPoint.any():
                        SVDescription += f"车辆{id(sv) % 1000}也在交叉路口，{self.getSVRelativeState(sv)}。 它的位置是({sv.position[0]:.2f}, {sv.position[1]:.2f})，速度是{sv.speed:.2f}米/秒，加速度是{sv.action['acceleration']:.2f}米/秒^2。潜在的碰撞点是({collisionPoint[0]:.2f}, {collisionPoint[1]:.2f})."
                    else:
                        SVDescription += f"车辆{id(sv) % 1000}也在交叉路口，{self.getSVRelativeState(sv)}。它的位置是({sv.position[0]:.2f}, {sv.position[1]:.2f})，速度是{sv.speed:.2f}米/秒，加速度是{sv.action['acceleration']:.2f}米/秒^2。您和它两辆车之间没有潜在的碰撞风险。"
                elif lidx == nextLane:
                    collisionPoint = self.getCollisionPoint(sv)
                    if collisionPoint:
                        SVDescription += f"车辆{id(sv) % 1000} 与您在同一车道，{self.getSVRelativeState(sv)}。它的位置是({sv.position[0]:.2f}, {sv.position[1]:.2f})，速度是{sv.speed:.2f}米/秒，加速度是{sv.action['acceleration']:.2f}米/秒^2。潜在的碰撞点是({collisionPoint[0]:.2f}, {collisionPoint[1]:.2f})."
                    else:
                        SVDescription += f"车辆{id(sv) % 1000} 与您在同一车道，{self.getSVRelativeState(sv)}。它的位置是({sv.position[0]:.2f}, {sv.position[1]:.2f})，速度是{sv.speed:.2f}米/秒，加速度是{sv.action['acceleration']:.2f}米/秒^2。您和它两辆车之间没有潜在的碰撞风险。"
                if self.isInDangerousArea(sv):
                    print(f"车辆{id(sv) % 1000}处在危险区域。")
                    SVDescription += f"车辆{id(sv) % 1000}也在交叉路口，{self.getSVRelativeState(sv)}。它的位置是({sv.position[0]:.2f}, {sv.position[1]:.2f})，速度是{sv.speed:.2f}米/秒，加速度是{sv.action['acceleration']:.2f}米/秒^2。这辆车位于您的视野范围内，在做决策时您需要关注它的状态。"
                else:
                    continue
            if SVDescription:
                descriptionPrefix = "周围还有其他车辆在行驶，以下是它们的基本信息："
                return descriptionPrefix + SVDescription
            else:
                '周围没有其他车辆行驶，所以您可以完全根据自己的想法驾驶。'
                return SVDescription

    def describe(self, decisionFrame: int) -> str:
        surroundVehicles = self.getSurrendVehicles(10)
        currentLaneIndex: LaneIndex = self.ego.lane_index
        if self.isInJunction(self.ego):
            roadCondition = "您在交叉路口驾驶，无法变道。"
            roadCondition += f"您的当前位置是({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})，速度是{self.ego.speed:.2f}米/秒，加速度是{self.ego.action['acceleration']:.2f}米/秒^2。"
            SVDescription = self.describeSVJunctionLane(currentLaneIndex)
        else:
            roadCondition = self.processNormalLane(currentLaneIndex)
            SVDescription = self.describeSVNormalLane(currentLaneIndex)

        return roadCondition + SVDescription


"""
import gymnasium as gym
import yaml
import warnings
warnings.filterwarnings("ignore")
config = yaml.load(open(r'D:\WhiteWing\DiLu\config.yaml'), Loader=yaml.FullLoader)
env_config = {
        'racetrack-v0':
        {
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,
            "collision_reward": -1,
            "lane_centering_cost": 4,
            "action_reward": -0.3,
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False
        }
}
episode = 0
envType = 'racetrack-v0'
env = gym.make(envType, render_mode="rgb_array")
env.configure(env_config[envType])
obs, info = env.reset()
for _ in range(10):
    print(env.action_space)
    action = {1,1}
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    sce = EnvScenario(env,envType,3)

    print(sce.getSurrendVehicles(3))
    print(sce.getClosestSV(sce.getSurrendVehicles(3)))
    print(sce.getLanePosition(sce.getSurrendVehicles(3)[0]))
    print(sce.getSVRelativeState(sce.getSurrendVehicles(3)[0]))
    print(sce.describe(3))
    episode+=1
    keyboard = input("请输入任何键")
"""

