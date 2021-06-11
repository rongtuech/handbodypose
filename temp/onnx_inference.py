import onnxruntime

class PoseDetectionONNX:
    def __init__(self, onnx_path):
        self.ort_session = onnxruntime.InferenceSession(onnx_path)

    def inference(self, input):
        # expand batch size
        input.unsqueeze_(0)
        # init input
        ort_inputs = {self.ort_session.get_inputs()[0].name: input}
        # inference
        pafs, heatmaps = self.ort_session.run(None, ort_inputs)

        return pafs, heatmaps