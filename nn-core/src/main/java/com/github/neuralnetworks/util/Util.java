package com.github.neuralnetworks.util;

import java.text.NumberFormat;
import java.util.Collection;
import java.util.Map.Entry;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.OutputError;

/**
 * Util class
 */
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
		sb.append(System.getProperty("line.separator")).append(nn.getClass().getSimpleName());
		break;
	    case Constants.OUTPUT_ERROR:
		OutputError error = (OutputError) e.getValue();
		NumberFormat n = NumberFormat.getInstance();
		n.setMaximumFractionDigits(5);
		sb.append(error.getClass().getSimpleName()).append(" - ").append(n.format(error.getTotalNetworkError()));
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
    
    public static void fillArray(final int[] array, final int value) {
	int len = array.length;
	if (len > 0) {
	    array[0] = value;
	}
	
	for (int i = 1; i < len; i += i) {
	    System.arraycopy(array, 0, array, i, ((len - i) < i) ? (len - i) : i);
	}
    }

    public static Layer getOppositeLayer(Connections connection, Layer layer) {
	return connection.getInputLayer() != layer ? connection.getInputLayer() : connection.getOutputLayer();
    }

    /**
     * @param layer
     * @return whether layer is in fact bias layer
     */
    public static boolean isBias(Layer layer) {
	return layer.getNeuronCount() == 1 && layer.getConnections().size() == 1 && layer.getConnections().iterator().next().getInputLayer() == layer;
    }

    public static boolean hasBias(Collection<Connections> connections) {
	for (Connections c : connections) {
	    if (Util.isBias(c.getInputLayer())) {
		return true;
	    }
	}

	return false;
    }
}
