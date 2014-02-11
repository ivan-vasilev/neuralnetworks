package com.github.neuralnetworks.test;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.ValuesProvider;

public class GeneralTest {

    @Test
    public void testValuesProvider() {
	ValuesProvider vp = new ValuesProvider();

	NeuralNetworkImpl nn = new NeuralNetworkImpl();;
	Layer i = new Layer();
	Layer h = new Layer();
	Layer o = new Layer();

	nn.addLayer(i);

	NNFactory.addFullyConnectedLayer(nn, h, 2, 3, true);
	NNFactory.addFullyConnectedLayer(nn, o, 4, 1, true);
	
	Matrix im = new Matrix(2, 2);
	vp.addValues(i, im);
	Matrix hm1 = vp.getValues(h, 3);
	Matrix hm2 = new Matrix(4, 2);
	vp.addValues(h, hm2);

	Matrix om = vp.getValues(o);

	assertTrue(im == vp.getValues(i, 2));
	assertTrue(im == vp.getValues(i));
	assertTrue(hm1 == vp.getValues(h, 3));
	assertTrue(hm2 == vp.getValues(h, 4));
	assertTrue(hm1 == vp.getValues(h, nn.getConnection(i, h)));
	assertTrue(hm2 == vp.getValues(h, nn.getConnection(h, o)));
	assertTrue(om == vp.getValues(o, 1));
	assertTrue(om == vp.getValues(o));
    }
}
