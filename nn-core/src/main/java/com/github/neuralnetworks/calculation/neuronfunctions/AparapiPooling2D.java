package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.Pooling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Environment;

/**
 * Base Aparapi connection calculator for pooling layers
 */
public class AparapiPooling2D extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = 8931101094464503687L;

    protected int inputColumns;
    protected int outputColumns;
    protected int outputLength;
    protected int regionLength;
    protected float[] input;
    protected float[] output;
    protected int[] featureMapOffsets;
    protected Pooling2DConnection current;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Pooling2DConnection c = (Pooling2DConnection) connections.keySet().iterator().next();

	this.init(c, connections.values().iterator().next(), output);

	this.execute(targetLayer.getNeuronCount());
    }

    protected void init(Pooling2DConnection c, Matrix input, Matrix output) {
	if (c != current) {
	    current = c;

	    ConvGridLayer inputLayer = (ConvGridLayer) c.getInputLayer();
	    ConvGridLayer outputLayer = (ConvGridLayer) c.getOutputLayer();
	    this.input = input.getElements();
	    this.output = output.getElements();
	    this.inputColumns = inputLayer.getColumns();
	    this.outputColumns = outputLayer.getColumns();
	    this.outputLength = outputLayer.getColumns() * outputLayer.getRows();
	    this.regionLength = c.getPoolingRegionRows() * c.getPoolingRegionCols();
	    
	    if (this.featureMapOffsets == null || this.featureMapOffsets.length != regionLength) {
		this.featureMapOffsets = new int[regionLength];
	    }
	    
	    for (int i = 0, j = 0; j < c.getPoolingRegionRows(); j++) {
		for (int k = 0; k < c.getPoolingRegionCols(); k++, i++) {
		    featureMapOffsets[i] = j * inputLayer.getColumns() + k;
		}
	    }
	}

	setExecutionMode(Environment.getInstance().getExecutionMode());
    }

    @Override
    public void run() {
	// to be implemented by subclasses
    }
}
