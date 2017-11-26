package com.github.neuralnetworks.analyzing;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.LayerOrderStrategy;
import com.github.neuralnetworks.calculation.operations.cpu.ConstantConnectionCalculator;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.UniqueList;

/**
 * show the size of each connection and calculate the min, max and avg weight
 * 
 * @author tmey
 */
public class ConnectionAnalysis
{


	/**
	 * only for feedforward networks
	 *
	 * @param neuralNetwork
	 * @return
	 */
	public static String analyseConnectionWeights(NeuralNetworkImpl neuralNetwork)
	{
		StringBuilder txtOutput = new StringBuilder();

		Set<Layer> layers = new UniqueList<Layer>();
		layers.addAll(neuralNetwork.getLayers());
		Layer inputLayer = neuralNetwork.getInputLayer();


		layers.remove(inputLayer);

		List<LayerOrderStrategy.ConnectionCandidate> connectionCandidates = extractConnection(neuralNetwork);

		for (LayerOrderStrategy.ConnectionCandidate connectionCandidate : connectionCandidates)
		{
			// analyse connections

			Connections connection = connectionCandidate.connection;

			// analyse connections
			if (connection instanceof Subsampling2DConnection)
			{
				if (connection.getOutputLayer().getName() != null)
				{
					txtOutput.append(connection.getOutputLayer().getName()).append(" (Subsampling Layer)\n");
				} else
				{
					txtOutput.append("Subsampling Layer\n");
				}
			} else if (connection instanceof Conv2DConnection)
			{
				if (connection.getOutputLayer().getName() != null)
				{
					txtOutput.append(connection.getOutputLayer().getName()).append(" (Convolution Layer)\n");
				} else
				{
					txtOutput.append("Convolution Layer\n");
				}

				Conv2DConnection conv2DConnection = (Conv2DConnection) connection;

				txtOutput.append(analyseConnections(conv2DConnection));

			} else if (connection instanceof FullyConnected)
			{
				if (connection.getOutputLayer().getName() != null)
				{
					txtOutput.append(connection.getOutputLayer().getName()).append(" (FullyConnected Layer)\n");
				} else
				{
					txtOutput.append("FullyConnected Layer\n");
				}

				FullyConnected fullyConnected = (FullyConnected) connection;

				txtOutput.append(analyseConnections(fullyConnected));
			} else
			{
				if (connection.getOutputLayer().getName() != null)
				{
					txtOutput.append(connection.getOutputLayer().getName()).append(" (Unknown Layer)\n");
				} else
				{
					txtOutput.append("Unknown Layer\n");
				}
			}


		}

		return txtOutput.toString();
	}

	private static List<LayerOrderStrategy.ConnectionCandidate> extractConnection(NeuralNetworkImpl neuralNetwork)
    {

        Set<Layer> calculatedLayers = new HashSet<Layer>();
        calculatedLayers.add(neuralNetwork.getInputLayer());

        neuralNetwork.getLayers().stream().filter(l -> ((LayerCalculatorImpl)neuralNetwork.getLayerCalculator()).getConnectionCalculator(l) instanceof ConstantConnectionCalculator).forEach(l -> {
            calculatedLayers.add(l);
        });

        List<LayerOrderStrategy.ConnectionCandidate> ccc = new BreadthFirstOrderStrategy(neuralNetwork, neuralNetwork.getOutputLayer(), calculatedLayers).order();

        Collections.reverse(ccc);
        return ccc;
    }

	private static String analyseConnections(Conv2DConnection connection)
	{
		StringBuilder txt = new StringBuilder();

		txt
				.append(" input filter: ").append(connection.getInputFilters())
				.append(" output filter: ").append(connection.getOutputFilters())
				.append(" filter colomns: ").append(connection.getFilterColumns())
				.append(" filter rows: ").append(connection.getFilterRows())
				.append(" column stride: ").append(connection.getColumnStride())
				.append(" row stride: ").append(connection.getRowStride())
				.append(" feature maps columns: ").append(connection.getInputFeatureMapColumns())
				.append(" feature maps rows: ").append(connection.getInputFeatureMapRows())
				.append(" feature maps length: ").append(connection.getInputFeatureMapLength())
				.append(" input: ").append(connection.getInputUnitCount())
				.append(" output: ").append(connection.getOutputUnitCount())

				.append("\n");


		Tensor weights = connection.getWeights();

		float min = Float.MAX_VALUE;
		float max = -Float.MAX_VALUE;
		float sum = 0;

		for (int i = weights.getStartIndex(); i < weights.getEndIndex(); i++)
		{
			float value = weights.getElements()[i];
			min = Math.min(value, min);
			max = Math.max(value, max);
			sum += value;
		}

		txt.append("min: ").append(min).append("\n")
				.append("max: ").append(max).append("\n")
				.append("avg: ").append((sum / weights.getSize())).append("\n");

		return txt.toString();
	}

	private static String analyseConnections(FullyConnected connection)
	{
		StringBuilder txt = new StringBuilder();
		txt.append("input: ").append(connection.getInputUnitCount())
				.append(" output: ").append(connection.getOutputUnitCount())
				.append("\n");

		Tensor weights = connection.getWeights();

		float min = Float.MAX_VALUE;
		float max = -Float.MAX_VALUE;
		float sum = 0;

		for (int i = weights.getStartIndex(); i < weights.getEndIndex(); i++)
		{
			float value = weights.getElements()[i];
			min = Math.min(value, min);
			max = Math.max(value, max);
			sum += value;
		}

		txt.append("min: ").append(min).append("\n")
				.append("max: ").append(max).append("\n")
				.append("avg: ").append((sum / weights.getSize())).append("\n");

		return txt.toString();
	}

}
