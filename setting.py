from enum import Enum


class TypeData(Enum):
    BODY = 0
    HAND = 1


class HandJointType(Enum):
    BAMB_0 = 0
    BAMB_1 = 1
    BIG_TOE = 2
    BIG_TOE_1 = 3
    BIG_TOE_2 = 4
    FINGER_1 = 5
    FINGER_1_1 = 6
    FINGER_1_2 = 7
    FINGER_1_3 = 8
    FINGER_2 = 9
    FINGER_2_1 = 10
    FINGER_2_2 = 11
    FINGER_2_3 = 12
    FINGER_3 = 13
    FINGER_3_1 = 14
    FINGER_3_2 = 15
    FINGER_3_3 = 16
    FINGER_4 = 17
    FINGER_4_1 = 18
    FINGER_4_2 = 19
    FINGER_4_3 = 20


class JointType(Enum):
    Nose = 0
    Neck = 1
    RightShoulder = 2
    RightElbow = 3
    RightHand = 4
    LeftShoulder = 5
    LeftElbow = 6
    LeftHand = 7
    RightWaist = 8
    RightKnee = 9
    RightFoot = 10
    LeftWaist = 11
    LeftKnee = 12
    LeftFoot = 13
    RightEye = 14
    LeftEye = 15
    RightEar = 16
    LeftEar = 17


coco_joint_indices= [
        JointType.Nose,
        JointType.LeftEye,
        JointType.RightEye,
        JointType.LeftEar,
        JointType.RightEar,
        JointType.LeftShoulder,
        JointType.RightShoulder,
        JointType.LeftElbow,
        JointType.RightElbow,
        JointType.LeftHand,
        JointType.RightHand,
        JointType.LeftWaist,
        JointType.RightWaist,
        JointType.LeftKnee,
        JointType.RightKnee,
        JointType.LeftFoot,
        JointType.RightFoot
    ]


LIMBS = [[JointType.Neck, JointType.RightWaist],
        [JointType.RightWaist, JointType.RightKnee],
        [JointType.RightKnee, JointType.RightFoot],
        [JointType.Neck, JointType.LeftWaist],
        [JointType.LeftWaist, JointType.LeftKnee],
        [JointType.LeftKnee, JointType.LeftFoot],
        [JointType.Neck, JointType.RightShoulder],
        [JointType.RightShoulder, JointType.RightElbow],
        [JointType.RightElbow, JointType.RightHand],
        [JointType.RightShoulder, JointType.RightEar],
        [JointType.Neck, JointType.LeftShoulder],
        [JointType.LeftShoulder, JointType.LeftElbow],
        [JointType.LeftElbow, JointType.LeftHand],
        [JointType.LeftShoulder, JointType.LeftEar],
        [JointType.Neck, JointType.Nose],
        [JointType.Nose, JointType.RightEye],
        [JointType.Nose, JointType.LeftEye],
        [JointType.RightEye, JointType.RightEar],
        [JointType.LeftEye, JointType.LeftEar]]


HANDLINES = [
    [HandJointType.BAMB_0, HandJointType.BAMB_1],
    [HandJointType.BAMB_1, HandJointType.BIG_TOE],
    [HandJointType.BIG_TOE, HandJointType.BIG_TOE_1],
    [HandJointType.BIG_TOE_1, HandJointType.BIG_TOE_2],
    [HandJointType.BAMB_0, HandJointType.FINGER_1],
    [HandJointType.FINGER_1, HandJointType.FINGER_1_1],
    [HandJointType.FINGER_1_1, HandJointType.FINGER_1_2],
    [HandJointType.FINGER_1_2, HandJointType.FINGER_1_3],
    [HandJointType.BAMB_0, HandJointType.FINGER_2],
    [HandJointType.FINGER_2, HandJointType.FINGER_2_1],
    [HandJointType.FINGER_2_1, HandJointType.FINGER_2_2],
    [HandJointType.FINGER_2_2, HandJointType.FINGER_2_3],
    [HandJointType.BAMB_0, HandJointType.FINGER_3],
    [HandJointType.FINGER_3, HandJointType.FINGER_3_1],
    [HandJointType.FINGER_3_1, HandJointType.FINGER_3_2],
    [HandJointType.FINGER_3_2, HandJointType.FINGER_3_3],
    [HandJointType.BAMB_0, HandJointType.FINGER_4],
    [HandJointType.FINGER_4, HandJointType.FINGER_4_1],
    [HandJointType.FINGER_4_1, HandJointType.FINGER_4_2],
    [HandJointType.FINGER_4_2, HandJointType.FINGER_4_3],
]