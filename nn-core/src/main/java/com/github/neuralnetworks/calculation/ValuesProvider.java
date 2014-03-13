package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * Provides Matrix instances for the layers of the network. It ensures that the
 * instances are reused.
 */
public class ValuesProvider implements Serializable {

    private static final long serialVersionUID = 1L;

    private int columns;
    private Map<Layer, Set<Matrix>> values;

    public ValuesProvider() {
	super();
	values = new HashMap<>();
    }

    /**
     * @return Matrix for connections. The connections must have a common layer and they must have the same dimensions.
     */
    public Matrix getValues(Layer targetLayer, Collection<Connections> connections) {
	return getValues(targetLayer, getUnitCount(targetLayer, connections));
    }

    /**
     * @return Matrix for connections. The connections must have a common layer and they must have the same dimensions.
     */
    public Matrix getValues(Layer targetLayer, Connections c) {
	return getValues(targetLayer, Arrays.asList(new Connections[] {c}));
    }

    /**
     * @return Matrix for layer. Works only in the case when the layer has only one associated matrix.
     */
    public Matrix getValues(Layer targetLayer) {
	return getValues(targetLayer, targetLayer.getConnections());
    }

    /**
     * Get values for layer based on provided dimensions
     * @param targetLayer
     * @param rows
     * @return
     */
    public Matrix getValues(Layer targetLayer, int rows) {
	if (!values.containsKey(targetLayer)) {
	    values.put(targetLayer, new HashSet<Matrix>());
	}

	Set<Matrix> set = values.get(targetLayer);
	Matrix result = set.stream().filter(m -> m.getRows() == rows && m.getColumns() == getColumns()).findFirst().orElse(null);

	if (result == null) {
	    set.add(result = new Matrix(rows, getColumns()));
	}

	return result;
    }

    public int getUnitCount(Layer targetLayer, Collection<Connections> connections) {
	int result = 0;
	for (Connections c : connections) {
	    if (c.getInputLayer() == targetLayer) {
		if (result == 0) {
		    result = c.getInputUnitCount();
		}

		if (result != c.getInputUnitCount()) {
		    throw new IllegalArgumentException("Some connections require different unit count");
		}
	    } else if (c.getOutputLayer() == targetLayer) {
		if (result == 0) {
		    result = c.getOutputUnitCount();
		}

		if (result != c.getOutputUnitCount()) {
		    throw new IllegalArgumentException("Some connections require different unit count");
		}
	    } else {
		throw new IllegalArgumentException("A connection doesn't have the targetLayer as either input or output");
	    }
	}

	return result;
    }

    public int getUnitCount(Layer targetLayer, Connections c) {
	Set<Connections> cs = new HashSet<>();
	cs.add(c);
	return getUnitCount(targetLayer, cs);
    }

    public void addValues(Layer l, Matrix m) {
	Set<Matrix> set = values.get(l);
	if (set == null) {
	    values.put(l, set = new HashSet<Matrix>());
	} else {
	    set.removeIf(o -> o.getRows() == m.getRows());
	}

	setColumns(m.getColumns());
	set.add(m);
    }

    public int getColumns() {
	if (columns == 0) {
	    values.values().forEach(v -> v.stream().filter(m -> columns < m.getColumns()).forEach(m -> columns = m.getColumns()));
	}

	return columns;
    }

    public void setColumns(int columns) {
	this.columns = columns;
    }
}
