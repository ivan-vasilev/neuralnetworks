package com.github.neuralnetworks.calculation.neuronfunctions;

import java.io.Serializable;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Environment;

/**
 * Base class for convolutional operations (2d)
 * This connection accept as input a single training example (as opposed to the weighted sum which works with multiple).
 *
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 */
public abstract class AparapiConv2D extends Kernel implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * input column count (columns of image for example)
     */
    protected final int inputColumns;

    /**
     * output column count (columns of image for example)
     */
    protected final int outputColumns;

    /**
     * number of samples per calculation (for example number of images)
     */
    protected final int miniBatchSize;

    /**
     * output columns * output rows
     */
    protected final int outputFeatureMapLength;
    
    /**
     * weights for single feature map
     */
    protected final int featureMapWeights;

    /**
     * input offset for each feature map in respect to the start index
     */
    //@Local TODO
    protected final int[] featureMapOffsets;

    /**
     * input
     */
    protected float[] input;

    /**
     * output
     */
    protected float[] output;

    /**
     * combined feature weights of all feature maps
     */
    //@Local TODO
    protected float[] weights;

    public AparapiConv2D(Conv2DConnection c, int miniBatchSize) {
	super();

	ConvGridLayer inputLayer = (ConvGridLayer) c.getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) c.getOutputLayer();

	this.weights = c.getWeights();
	this.miniBatchSize = miniBatchSize;
	this.inputColumns = inputLayer.getFeatureMapColumns();
	this.outputColumns = outputLayer.getFeatureMapColumns();
	this.outputFeatureMapLength = outputLayer.getFeatureMapLength();
	this.featureMapWeights = c.getWeights().length / outputLayer.getFilters();
	this.featureMapOffsets = new int[featureMapWeights];

	for (int i = 0, offset = 0; i < inputLayer.getFilters(); i++) {
	    for (int j = 0; j < c.getKernelRows(); j++) {
		for (int k = 0; k < c.getKernelColumns(); k++) {
		    featureMapOffsets[offset++] = i * inputLayer.getFeatureMapLength() + j * inputLayer.getFeatureMapColumns() + k;
		}
	    }
	}
    }

    public void calculate(Conv2DConnection c, Matrix input, Matrix output) {
	if (c != null) {
	    init(c, input, output);
	    Environment.getInstance().getExecutionStrategy().execute(this, output.getRows());
	}
    }

    /**
     * Converts connection, input and output data to one dimensional arrays (because of the Aparapi limitations)
     */
    protected void init(Conv2DConnection c, Matrix input, Matrix output) {
	this.input = input.getElements();
	this.output = output.getElements();
    }

    @Override
    public void run() {
	int id = getGlobalId();

	conv(featureMapWeights * (id / outputFeatureMapLength), ((id % outputFeatureMapLength) / outputColumns) * inputColumns + id % outputColumns);
    }

    /**
     * the actual convolution
     * @param weightsStartId
     * @param inputStartId
     */
    protected void conv(int weightsStartId, int inputStartId) {
    }
}
