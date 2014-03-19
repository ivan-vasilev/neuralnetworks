package com.github.neuralnetworks.util;



/**
 * Simple matrix representation with one-dimensional array. This is required,
 * because Aparapi supports only one-dim arrays (otherwise the execution is
 * transferred to the cpu)
 */
public class Matrix extends Tensor {

    private static final long serialVersionUID = 1L;

    public Matrix() {
	super();
    }

    public Matrix(float[] elements, int columns) {
	super(elements, elements.length / columns, columns);
    }

    public Matrix(int rows, int columns) {
	super(rows, columns);
    }

    public Matrix(Matrix copy) {
	super(copy.getRows(), copy.getColumns());
    }

    public Matrix(Tensor parent, int[] dimStart, int[] dimEnd) {
	super(parent, dimStart, dimEnd);
    }

    public int getColumns() {
	return getDimensionLength(getColumnsDimension());
    }

    protected int getColumnsDimension() {
	int d = getRowsDimension() + 1;
	for (int i = dimensions.length - 1; i > 0; i--) {
	    if (dimStart[i] != dimEnd[i] && d < i) {
		d = i;
		break;
	    }
	}

	return d;
    }
    
    public int getColumnElementsDistance() {
	return getDimensionElementsDistance(getColumnsDimension());
    }

    public int getColumnsStartIndex() {
	return getStartIndex(getColumnsDimension());
    }

    public int getRows() {
	return getDimensionLength(getRowsDimension());
    }

    public int getRowElementsDistance() {
	return getDimensionElementsDistance(getRowsDimension());
    }

    public int getRowsStartIndex() {
	return getStartIndex(getRowsDimension());
    }

    protected int getRowsDimension() {
	int d = dimensions.length - 2;
	for (int i = 0; i < dimensions.length - 1; i++) {
	    if (dimStart[i] != dimEnd[i]) {
		d = i;
		break;
	    }
	}

	return d;
    }

    /* (non-Javadoc)
     * d[0] - rows
     * d[1] - columns
     * @see com.github.neuralnetworks.util.Tensor#get(int[])
     */
    @Override
    public float get(int... d) {
	if (d.length != 2) {
	    throw new IllegalArgumentException("Please provide row and column only");
	}

	Util.fillArray(dimTmp, 0);
	dimTmp[getRowsDimension()] = d[0];
	dimTmp[getColumnsDimension()] = d[1];

	return super.get(dimTmp);
    }

    @Override
    public void set(float value, int... d) {
	if (d.length != 2) {
	    throw new IllegalArgumentException("Please provide row and column only");
	}

	Util.fillArray(dimTmp, 0);
	dimTmp[getRowsDimension()] = d[0];
	dimTmp[getColumnsDimension()] = d[1];

	super.set(value, dimTmp);
    }
}
