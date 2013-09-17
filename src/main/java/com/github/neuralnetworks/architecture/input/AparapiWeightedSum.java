package com.github.neuralnetworks.architecture.input;

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
	public void calculateForward(IConnections graph, float[] inputValues, float[] result) {
		ConnectionGraph cg = graph.getConnectionGraph();
		final float weights[] = cg.getWeights();

		final int startNeuronIndex = graph.getOutputLayerStartNeuron();

		Kernel kernel = new Kernel() {
			@Override
			public void run() {
				int id = getGlobalId();
//				for (int i = 0; i < layerValues.length; i++) {
//					//result[startNeuronIndex + id] += layerValues[]
//				}
			}
		};
		//kernel.execute(graph.getOutputLayerNeuronsCount());
	}

	@Override
	public void calculateBackward(IConnections graph, float[] inputValues, float[] result) {
		// TODO Auto-generated method stub

	}
}
