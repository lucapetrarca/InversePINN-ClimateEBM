# InversePINN-ClimateEBM
Objetivo Teórico: Demostrar analíticamente la existencia de perfiles de temperatura de equilibrio global en el modelo climático de Balance de Energía (EBM) 1D mediante el uso de sub/supersoluciones y el Teorema de Punto Fijo de Schauder.

Objetivo Topológico: Analizar la estabilidad estructural del clima frente a variaciones en la radiación solar ($Q$), utilizando el Grado de Leray-Schauder para justificar matemáticamente las bifurcaciones y los saltos catastróficos.

Objetivo Computacional: Implementar una Physics-Informed Neural Network en modo inverso (Inverse PINN) para descubrir, a partir de datos de temperatura observados, los parámetros físicos ocultos del sistema (como el coeficiente de difusión de calor oceánico y la función de albedo latitudinal) que sostienen dicho equilibrio topológico.