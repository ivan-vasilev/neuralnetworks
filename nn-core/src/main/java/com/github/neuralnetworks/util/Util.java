package com.github.neuralnetworks.util;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.OutputError;

public class Util {

    /**
     * @param properties
     * @return convert properties to string (for printing purposes for example)
     */
    public static String propertiesToString(Properties properties) {
	StringBuilder sb = new StringBuilder();
	for (Entry<String, Object> e : properties.entrySet()) {
	    sb.append(e.getKey()).append(": ");
	    switch (e.getKey()) {
	    case Constants.NEURAL_NETWORK:
		NeuralNetwork nn = (NeuralNetwork) e.getValue();
		sb.append(System.getProperty("line.separator")).append(nn.getClass().getSimpleName()).append( " layers:").append(System.getProperty("line.separator"));
		for (Layer l : nn.getLayers()) {
		    sb	.append(l.getNeuronCount()).append(" neurons; ")
		    	.append(l.getForwardInputFunction().getClass().getSimpleName()).append(" forward input; ")
		    	.append(l.getBackwardInputFunction().getClass().getSimpleName()).append(" backward input; ")
		    	.append(l.getActivationFunction().getClass().getSimpleName()).append(" activation function; ")
		    	.append(System.getProperty("line.separator"));
		}
		break;
	    case Constants.LAYERS:
		@SuppressWarnings("unchecked")
		List<Integer> layers = (List<Integer>) e.getValue();
		sb.append(Arrays.toString(layers.toArray()));
		break;
	    case Constants.OUTPUT_ERROR:
		OutputError error = (OutputError) e.getValue();
		NumberFormat n = NumberFormat.getInstance();
		n.setMaximumFractionDigits(5);
		sb.append(error.getClass().getCanonicalName()).append(" - ").append(n.format(error.getTotalNetworkError()));
		break;
	    default:
		sb.append(e.getValue());
	    }

	    sb.append(System.getProperty("line.separator"));
	}

	return sb.toString();
    }

    public static void fillArray(final float[] array, final float value) {
	int len = array.length;
	if (len > 0) {
	    array[0] = value;
	}

	for (int i = 1; i < len; i += i) {
	    System.arraycopy(array, 0, array, i, ((len - i) < i) ? (len - i) : i);
	}
    }
}
