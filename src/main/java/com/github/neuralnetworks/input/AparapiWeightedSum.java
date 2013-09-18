package com.github.neuralnetworks.input;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.ConnectionGraph;
import com.github.neuralnetworks.architecture.IConnections;

/**
 * weighted sum input function
 *
 * @author hok
 *
 */
public class AparapiWeightedSum implements InputFunction {

	private static final long serialVersionUID = 8650655018964028006L;

	@Override
	public void calculateForward(IConnections graph, final float[] inputValues, final float[] result) {
		ConnectionGraph cg = graph.getConnectionGraph();
		final float weights[] = cg.getWeights();
		final int neuronWeightsCount = cg.getNeuronWeightsCount();
		final int outputStartIndex = graph.getOutputLayerStartNeuron();
		final int inputStartIndex = graph.getInputLayerStartNeuron();

		Kernel kernel = new Kernel() {
			@Override
			public void run() {
				int id = getGlobalId();
				for (int i = 0; i < neuronWeightsCount; i++) {
					System.out.println(id);
					result[outputStartIndex + id] += inputValues[inputStartIndex + i] * weights[neuronWeightsCount * id + i];
				}
			}
		};
		int outputNeuronsCount = weights.length / cg.getNeuronWeightsCount();
		kernel.execute(outputNeuronsCount);
	}

	@Override
	public void calculateBackward(IConnections graph, final float[] inputValues, final float[] result) {
		ConnectionGraph cg = graph.getConnectionGraph();
		final float weights[] = cg.getWeights();
		final int neuronWeightsCount = cg.getNeuronWeightsCount();
		final int inputNeuronsCount = weights.length / neuronWeightsCount;
		final int outputStartIndex = graph.getOutputLayerStartNeuron();
		final int inputStartIndex = graph.getInputLayerStartNeuron();

		Kernel kernel = new Kernel() {
			@Override
			public void run() {
				int id = getGlobalId();
				for (int i = 0; i < inputNeuronsCount; i++) {
					result[inputStartIndex + id] += inputValues[outputStartIndex + i] * weights[neuronWeightsCount * i + id];
				}
			}
		};
		kernel.execute(cg.getNeuronWeightsCount());
	}
}
