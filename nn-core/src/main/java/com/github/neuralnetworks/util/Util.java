package com.github.neuralnetworks.util;

import java.util.Collection;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
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
	return layer.getNeuronCount() == 1 && layer.getConnections().size() == 1 && layer.getConnections().iterator().next().getInputLayer() == layer;
    }

    /**
     * @param layer
     * @return whether layer is in fact subsampling layer (based on the connections)
     */
    public static boolean isSubsampling(Layer layer) {
	if (layer instanceof ConvGridLayer) {
	    ConvGridLayer l = (ConvGridLayer) layer;
	    if (l.getConnections().size() == 1) {
		for (Connections c : l.getConnections()) {
		    if (c instanceof Subsampling2DConnection && c.getOutputLayer() == l) {
			return true;
		    }
		}
	    }
	}

	return false;
    }
    
    /**
     * @param layer
     * @return whether layer is in fact convolutional layer (based on the connections)
     */
    public static boolean isConvolutional(Layer layer) {
	if (layer instanceof ConvGridLayer) {
	    ConvGridLayer l = (ConvGridLayer) layer;
	    for (Connections c : l.getConnections()) {
		if (c instanceof Conv2DConnection && c.getOutputLayer() == l) {
		    return true;
		}
	    }
	}
	
	return false;
    }

    /**
     * @param connections
     * @return whether there is a bias connection in the list
     */
    public static boolean hasBias(Collection<Connections> connections) {
	for (Connections c : connections) {
	    if (Util.isBias(c.getInputLayer())) {
		return true;
	    }
	}

	return false;
    }
}
