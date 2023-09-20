# Dynamic Factor Analysis

`dynamicfactoranalysis` is a Python package that provides tools for dynamic factor analysis.

## Main Features

The current version contains the following implementation

- `DynamicPCA`

  - Implementation of dynamic principal component analysis following Forni et al. (2000), Forni et al. (2005), and Favero et al. (2005). See `examples/DynamicPCA`.
- `DynamicFactorModel`

  - Implementation of the dynamic form of dynamic factor model. The implementation is modified from the statsmodels implementation. See `examples/DynamicFactorModel`.
- `DynamicFactorModelOptimizer`

  - Implementation for dynamic factor model order selection following Bai & Ng (2002) and Bai & Ng (2008). See `examples/DynamicFactorModel`.

## Usage

Copy the folder `dynamicfactoranalysis` to your project directory and import the package.

## Contributing

Contributions are welcomed, especially in creating new models and features.

## License

MIT License
