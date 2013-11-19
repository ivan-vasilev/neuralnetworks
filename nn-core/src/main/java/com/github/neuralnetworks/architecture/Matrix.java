package com.github.neuralnetworks.architecture;

import java.io.Serializable;

/**
 * 
 * this is a graph representation based on a one dimensional array (in order to
 * facilitate gpu computations)
 * 
 */
public class Matrix implements Serializable {

    private static final long serialVersionUID = 1L;

    private float[] elements;
    private int columns;

    public Matrix() {
	super();
    }

    public Matrix(Matrix copy) {
	super();
	this.elements = new float[copy.elements.length];
	this.columns = copy.columns;
    }

    public Matrix(float[] elements, int columns) {
	super();
	this.elements = elements;
	this.columns = columns;
    }

    public Matrix(int rows, int columns) {
	super();
	this.elements = new float[rows * columns];
	this.columns = columns;
    }

    public float[] getElements() {
	return elements;
    }

    public void setElements(float[] elements) {
	this.elements = elements;
    }

    public int getColumns() {
	return columns;
    }

    public void setColumns(int columns) {
	this.columns = columns;
    }

    public int getRows() {
	return this.elements.length / this.columns;
    }

    public void set(int row, int column, float value) {
	elements[row * columns + column] = value;
    }

    public float get(int row, int column) {
	return elements[row * columns + column];
    }

    public int getColumn(int index) {
	return index % columns;
    }

    public int getRow(int index) {
	return index / columns;
    }
}
