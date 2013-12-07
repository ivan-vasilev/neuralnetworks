package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Environment;

/**
 * Base Aparapi connection calculator for subsampling layers. In order to work Aparapi has to use only one-dimensional arrays with simple data types and can only call member methods of the Kernel class itself
 * Aparapi also requires to use only float (or int) data types
 */
public class AparapiSubsampling2D extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = 8931101094464503687L;

    protected int inputLength;
    protected int outputLength;
    protected int regionLength;
    protected float[] input;
    protected float[] output;
    protected float[] currentValues;
    protected int[] featureMapOffsets;
    protected int[] outputInputIndexes;
    protected Subsampling2DConnection current;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Subsampling2DConnection c = (Subsampling2DConnection) connections.keySet().iterator().next();

	this.init(c, connections.values().iterator().next(), output);

	this.execute(targetLayer.getNeuronCount());
    }

    protected void init(Subsampling2DConnection c, Matrix input, Matrix output) {
	if (c != current) {
	    current = c;

	    ConvGridLayer inputLayer = (ConvGridLayer) c.getInputLayer();
	    ConvGridLayer outputLayer = (ConvGridLayer) c.getOutputLayer();
	    this.input = input.getElements();
	    this.output = output.getElements();
	    this.inputLength = inputLayer.getColumns() * inputLayer.getRows();
	    this.outputLength = outputLayer.getColumns() * outputLayer.getRows();
	    this.regionLength = c.getSubsamplingRegionRows() * c.getSubsamplingRegionCols();
	    this.featureMapOffsets = new int[regionLength];
	    this.outputInputIndexes = new int [outputLength];
	    this.currentValues = new float[regionLength];

	    for (int i = 0, j = 0; j < c.getSubsamplingRegionRows(); j++) {
		for (int k = 0; k < c.getSubsamplingRegionCols(); k++, i++) {
		    featureMapOffsets[i] = j * inputLayer.getColumns() + k;
		}
	    }

	    int inputRowsOffset = (input.getRows() % c.getSubsamplingRegionRows()) / 2;
	    int inputColsOffset = (input.getColumns() % c.getSubsamplingRegionCols()) / 2;
	    for (int j = 0; j < outputLayer.getRows(); j++) {
		for (int k = 0; k < outputLayer.getColumns(); k++) {
		    outputInputIndexes[j * inputLayer.getColumns() + k] = (inputRowsOffset + c.getSubsamplingRegionRows() * j) * inputLayer.getColumns() + inputColsOffset + c.getSubsamplingRegionCols() * k;
		}
	    }
	}

	setExecutionMode(Environment.getInstance().getExecutionMode());
    }

    /* (non-Javadoc)
     * @see com.amd.aparapi.Kernel#run()
     * currentValues array is populated here. The values from this array are the ones that take part in the pooling in currentValuesUpdated
     */
    @Override
    public void run() {
	int id = getGlobalId();

	// get feature map data
	int featureMap = id / outputLength;
	int inputStartIndex = featureMap * inputLength + outputInputIndexes[id - featureMap * outputLength];
	for (int i = 0; i < regionLength; i++) {
	    currentValues[i] = input[inputStartIndex + featureMapOffsets[i]];
	}

	currentValuesUpdated();
    }

    /**
     * This is where the subsampling happens based on the extracted values in currentValues
     */
    protected void currentValuesUpdated() {
    }
}
