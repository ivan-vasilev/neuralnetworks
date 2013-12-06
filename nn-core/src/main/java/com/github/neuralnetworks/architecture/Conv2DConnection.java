package com.github.neuralnetworks.architecture;

import java.util.List;

import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Convolutional connection between layers (for 2d input data)
 */
public class Conv2DConnection extends ConnectionsImpl {

    protected List<Matrix> filters;

    public Conv2DConnection(ConvGridLayer inputLayer, ConnectionCalculator convCalculator) {
	super(inputLayer, new ConvGridLayer(0, 0, 0, convCalculator));
    }

    public List<Matrix> getFilters() {
	return filters;
    }

    public void setFilters(List<Matrix> filters) {
	this.filters = filters;
    }

    public void addFilter(Matrix filter) {
	if (filters == null) {
	    filters = new UniqueList<>();
	}

	for (Matrix m : filters) {
	    if (filter.getColumns() != m.getColumns() || filter.getRows() != m.getRows()) {
		throw new IllegalArgumentException();
	    }
	}

	filters.add(filter);

	ConvGridLayer inputLayer = (ConvGridLayer) getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) getOutputLayer();
	outputLayer.setColumns(inputLayer.getColumns() - inputLayer.getColumns() % filter.getColumns());
	outputLayer.setRows(inputLayer.getRows() - inputLayer.getRows() % filter.getRows());
	outputLayer.setFilters(filters.size());
    }

    public void removeFeatureMap(Matrix featureMap) {
	if (filters != null) {
	    filters.remove(featureMap);
	    ConvGridLayer l = (ConvGridLayer) getOutputLayer();
	    l.setFilters(filters.size());
	}
    }
}
