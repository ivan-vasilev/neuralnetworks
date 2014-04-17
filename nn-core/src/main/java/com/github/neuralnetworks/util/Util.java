package com.github.neuralnetworks.util;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Collection;
import java.util.stream.IntStream;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;

/**
 * Util class
 */
public class Util {

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
	if (layer.getConnections().size() == 1) {
	    Connections c = layer.getConnections().get(0);
	    if (c.getInputLayer() == layer) {
		if (c instanceof Conv2DConnection) {
		    Conv2DConnection cc = (Conv2DConnection) c;
		    return cc.getInputFilters() == 1 && cc.getInputFeatureMapRows() == cc.getOutputFeatureMapRows() && cc.getInputFeatureMapColumns() == cc.getOutputFeatureMapColumns();
		} else if (c instanceof FullyConnected) {
		    FullyConnected cg = (FullyConnected) c;
		    return cg.getWeights().getColumns() == 1;
		}
	    }
	}

	return false;
    }

    /**
     * @param layer
     * @return whether layer is in fact subsampling layer (based on the
     *         connections)
     */
    public static boolean isSubsampling(Layer layer) {
	Conv2DConnection conv = null;
	Subsampling2DConnection ss = null;
	for (Connections c : layer.getConnections()) {
	    if (c instanceof Conv2DConnection) {
		conv = (Conv2DConnection) c;
	    } else if (c instanceof Subsampling2DConnection) {
		ss = (Subsampling2DConnection) c;
	    }
	}

	if (ss != null && (ss.getOutputLayer() == layer || conv == null)) {
	    return true;
	}

	return false;
    }

    /**
     * @param layer
     * @return whether layer is in fact convolutional layer (based on the
     *         connections)
     */
    public static boolean isConvolutional(Layer layer) {
	Conv2DConnection conv = null;
	Subsampling2DConnection ss = null;
	for (Connections c : layer.getConnections()) {
	    if (c instanceof Conv2DConnection) {
		conv = (Conv2DConnection) c;
	    } else if (c instanceof Subsampling2DConnection) {
		ss = (Subsampling2DConnection) c;
	    }
	}

	if (conv != null && (conv.getOutputLayer() == layer || ss == null)) {
	    return true;
	}

	return false;
    }

    /**
     * @param connections
     * @return whether there is a bias connection in the list
     */
    public static boolean hasBias(Collection<Connections> connections) {
	return connections.stream().filter(c -> isBias(c.getInputLayer())).findAny().isPresent();
    }

    public static void printMatrix(float[] array, int rows, int columns) {
	StringBuilder sb = new StringBuilder();
	NumberFormat formatter = new DecimalFormat("#0.00");

	IntStream.range(0, columns).forEach(i -> {
	    IntStream.range(0, columns).forEach(j -> sb.append(formatter.format(array[i * columns + j])).append(" "));
	    sb.append(System.getProperty("line.separator"));
	});

	System.out.println(sb.toString());
    }
}
