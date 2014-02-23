package com.github.neuralnetworks.calculation.neuronfunctions;

import java.io.Serializable;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Conv2DConnection;
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
     * stride
     */
    protected final int stride;

    /**
     * input offset for each feature map in respect to the start index
     */
    //@Local TODO
    @Constant
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
    protected final float[] weights;

    public AparapiConv2D(Conv2DConnection c, int miniBatchSize) {
	super();

	this.weights = c.getWeights();
	this.miniBatchSize = miniBatchSize;
	this.inputColumns = c.getInputFeatureMapColumns();
	this.outputColumns = c.getOutputFeatureMapColumns();
	this.outputFeatureMapLength = c.getOutputFeatureMapLength();
	this.stride = c.getStride();
	this.featureMapWeights = c.getWeights().length / c.getOutputFilters();
	this.featureMapOffsets = new int[featureMapWeights];

	for (int i = 0, offset = 0; i < c.getInputFilters(); i++) {
	    for (int j = 0; j < c.getKernelRows(); j++) {
		for (int k = 0; k < c.getKernelColumns(); k++) {
		    featureMapOffsets[offset++] = i * c.getInputFeatureMapLength() + j * c.getInputFeatureMapColumns() + k;
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

	conv(featureMapWeights * (id / outputFeatureMapLength), ((id % outputFeatureMapLength) / outputColumns) * inputColumns * stride + (id % outputColumns) * stride);
    }

    /**
     * the actual convolution
     * @param weightsStartId
     * @param inputStartId
     */
    protected void conv(int weightsStartId, int inputStartId) {
    }
}
