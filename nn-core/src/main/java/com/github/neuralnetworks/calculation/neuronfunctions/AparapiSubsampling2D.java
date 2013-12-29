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
 * Base Aparapi connection calculator for subsampling layers.
 * 
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 */
public class AparapiSubsampling2D extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = 8931101094464503687L;

    /**
     * Length of the input image (rows * cols)
     */
    protected int inputLength;

    /**
     * Length of the output image (rows * cols)
     */
    protected int outputLength;

    /**
     * Length of the subsampling region (subsampling rows *  subsampling cols)
     */
    protected int regionLength;

    /**
     * input data
     */
    protected float[] input;

    /**
     * output
     */
    protected float[] output;

    /**
     * Contains only the input values that are in the current region. The current region is determined by the current output neuron
     */
    protected float[] currentValues;

    /**
     * Contains the offset in the input array for each cell of the current region. The offset is calculated in respect to the first cell of the region
     */
    protected int[] featureMapOffsets;

    /**
     * Mapping between output-input indexes.
     */
    protected int[] outputInputIndexes;
    protected Subsampling2DConnection current;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Subsampling2DConnection c = (Subsampling2DConnection) connections.keySet().iterator().next();

	this.init(c, connections.values().iterator().next(), output);

	// the code is executed with as many kernels as the output layer neurons count
	this.execute(targetLayer.getNeuronCount());
    }

    /**
     * Populates featureMapOffsets and outputInputIndexes
     * @param c
     * @param input
     * @param output
     */
    protected void init(Subsampling2DConnection c, Matrix input, Matrix output) {
	if (c != current) {
	    current = c;

	    ConvGridLayer inputLayer = (ConvGridLayer) c.getInputLayer();
	    ConvGridLayer outputLayer = (ConvGridLayer) c.getOutputLayer();
	    this.input = input.getElements();
	    this.output = output.getElements();
	    this.inputLength = inputLayer.getFeatureMapColumns() * inputLayer.getFeatureMapRows();
	    this.outputLength = outputLayer.getFeatureMapColumns() * outputLayer.getFeatureMapRows();
	    this.regionLength = c.getSubsamplingRegionRows() * c.getSubsamplingRegionCols();
	    this.featureMapOffsets = new int[regionLength];
	    this.outputInputIndexes = new int [outputLength];
	    this.currentValues = new float[regionLength];

	    for (int i = 0, j = 0; j < c.getSubsamplingRegionRows(); j++) {
		for (int k = 0; k < c.getSubsamplingRegionCols(); k++, i++) {
		    featureMapOffsets[i] = j * inputLayer.getFeatureMapColumns() + k;
		}
	    }

	    int inputRowsOffset = (input.getRows() % c.getSubsamplingRegionRows()) / 2;
	    int inputColsOffset = (input.getColumns() % c.getSubsamplingRegionCols()) / 2;
	    for (int j = 0; j < outputLayer.getFeatureMapRows(); j++) {
		for (int k = 0; k < outputLayer.getFeatureMapColumns(); k++) {
		    outputInputIndexes[j * inputLayer.getFeatureMapColumns() + k] = (inputRowsOffset + c.getSubsamplingRegionRows() * j) * inputLayer.getFeatureMapColumns() + inputColsOffset + c.getSubsamplingRegionCols() * k;
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

	// get current feature map
	int featureMap = id / outputLength;

	// calculate the start index (in the input of the current subsampling region)
	int inputStartIndex = featureMap * inputLength + outputInputIndexes[id - featureMap * outputLength];

	// populate currentValues based on the inputStartIndex and the offsets in featureMapOffsets
	for (int i = 0; i < regionLength; i++) {
	    currentValues[i] = input[inputStartIndex + featureMapOffsets[i]];
	}

	// the actual subsampling
	currentValuesUpdated();
    }

    /**
     * This is where the subsampling happens based on the extracted values in currentValues
     */
    protected void currentValuesUpdated() {
    }
}
