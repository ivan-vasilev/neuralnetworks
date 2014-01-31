package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticPooling2D;

/**
 * Factory for connections
 */
public class ConnectionFactory {

    public static FullyConnected fullyConnected(Layer inputLayer, Layer outputLayer) {
	return new FullyConnected(inputLayer, outputLayer);
    }

    public static Conv2DConnection convConnection(ConvGridLayer inputLayer, int kernelRows, int kernelColumns, int filters) {
	return new Conv2DConnection(inputLayer, kernelColumns, kernelRows, filters);
    }

    public static Conv2DConnection convSigmoidConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int kernelRows, int kernelColumns, int kernels) {
	Conv2DConnection result = convConnection(inputLayer, kernelRows, kernelColumns, kernels);
	lc.addConnectionCalculator(result.getOutputLayer(), new AparapiConv2DSigmoid());
	return result;
    }

    public static Conv2DConnection convSoftReLUConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int kernelRows, int kernelColumns, int kernels) {
	Conv2DConnection result = convConnection(inputLayer, kernelRows, kernelColumns, kernels);
	lc.addConnectionCalculator(result.getOutputLayer(), new AparapiConv2DSoftReLU());
	return result;
    }

    public static Conv2DConnection convTanhConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int kernelRows, int kernelColumns, int kernels) {
	Conv2DConnection result = convConnection(inputLayer, kernelRows, kernelColumns, kernels);
	lc.addConnectionCalculator(result.getOutputLayer(), new AparapiConv2DTanh());
	return result;
    }

    public static Subsampling2DConnection subsamplingConnection(ConvGridLayer inputLayer, int regionRows, int regionCols) {
	return new Subsampling2DConnection(inputLayer, regionRows, regionCols);
    }

    public static Subsampling2DConnection maxPoolingConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int regionRows, int regionCols, int miniBatchSize) {
	Subsampling2DConnection c = subsamplingConnection(inputLayer, regionRows, regionCols);
	lc.addConnectionCalculator(c.getOutputLayer(), new AparapiMaxPooling2D(c, miniBatchSize));

	return c;
    }

    public static Subsampling2DConnection averagePoolingConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int regionRows, int regionCols, int miniBatchSize) {
	Subsampling2DConnection c = subsamplingConnection(inputLayer, regionRows, regionCols);
	lc.addConnectionCalculator(c.getOutputLayer(), new AparapiAveragePooling2D(c, miniBatchSize));

	return c;
    }

    public static Subsampling2DConnection stochasticPoolingConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int regionRows, int regionCols, int miniBatchSize) {
	Subsampling2DConnection c = subsamplingConnection(inputLayer, regionRows, regionCols);
	lc.addConnectionCalculator(c.getOutputLayer(), new AparapiStochasticPooling2D(c, miniBatchSize));

	return c;
    }

    public static FullyConnected biasConnection(NeuralNetworkImpl nn, Layer targetLayer) {
	Layer biasLayer = new BiasLayer();
	nn.addLayer(biasLayer);
	return new FullyConnected(biasLayer, targetLayer);
    }
}
