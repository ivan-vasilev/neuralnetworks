package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Matrix;
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

    public static Conv2DConnection convConnection(ConvGridLayer inputLayer, int featureMapRows, int featureMapColumns, int featureMaps) {
	Conv2DConnection result = new Conv2DConnection(inputLayer);

	for (int i = 0; i < featureMaps; i++) {
	    result.addFilter(new Matrix(featureMapRows, featureMapColumns));
	}

	return result;
    }

    public static Conv2DConnection convSigmoidConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int featureMapRows, int featureMapColumns, int featureMaps) {
	Conv2DConnection result = convConnection(inputLayer, featureMapRows, featureMapColumns, featureMaps);
	lc.addConnectionCalculator(result.getOutputLayer(), new AparapiConv2DSigmoid());
	return result;
    }

    public static Conv2DConnection convSoftReLUConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int featureMapRows, int featureMapColumns, int featureMaps) {
	Conv2DConnection result = convConnection(inputLayer, featureMapRows, featureMapColumns, featureMaps);
	lc.addConnectionCalculator(result.getOutputLayer(), new AparapiConv2DSoftReLU());
	return result;
    }

    public static Conv2DConnection convTanhConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int featureMapRows, int featureMapColumns, int featureMaps) {
	Conv2DConnection result = convConnection(inputLayer, featureMapRows, featureMapColumns, featureMaps);
	lc.addConnectionCalculator(result.getOutputLayer(), new AparapiConv2DTanh());
	return result;
    }
    
    public static Subsampling2DConnection subsamplingConnection(ConvGridLayer inputLayer, int regionRows, int regionCols) {
	return new Subsampling2DConnection(inputLayer, regionRows, regionCols);
    }

    public static Subsampling2DConnection maxPoolingConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int regionRows, int regionCols) {
	Subsampling2DConnection c = subsamplingConnection(inputLayer, regionRows, regionCols);
	lc.addConnectionCalculator(c.getOutputLayer(), new AparapiMaxPooling2D());

	return c;
    }
    
    public static Subsampling2DConnection averagePoolingConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int regionRows, int regionCols) {
	Subsampling2DConnection c = subsamplingConnection(inputLayer, regionRows, regionCols);
	lc.addConnectionCalculator(c.getOutputLayer(), new AparapiAveragePooling2D());

	return c;
    }
    
    public static Subsampling2DConnection stochasticPoolingConnection(ConvGridLayer inputLayer, LayerCalculatorImpl lc, int regionRows, int regionCols) {
	Subsampling2DConnection c = subsamplingConnection(inputLayer, regionRows, regionCols);
	lc.addConnectionCalculator(c.getOutputLayer(), new AparapiStochasticPooling2D());

	return c;
    }
}
