"""
Arquitectura Custom CNN: 'UAVShipNet'
Disenada desde cero en PyTorch especificamente para clasificacion de barcos
en imagenes aereas satelitales (80x80 px RGB) embarcada en UAV/Drones.
Sustenta los requisitos de Nivel 5 de ABET (Criterio C1 y C2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Bloque convolucional doble con Batch Normalization, activacion LeakyReLU
    y Dropout espacial para regularizacion robusta.
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x

class UAVShipNet(nn.Module):
    """
    Red Neuronal Convolucional compacta y de alta precision:
    - Entrada: (B, 3, 80, 80)
    - 4 Bloques Convolucionales progresivos (32 -> 64 -> 128 -> 256 filtros)
    - Global Average Pooling (GAP) para reduccion de parametros e invariancia espacial
    - Cabeza de clasificacion regularizada para inferencia binaria ultra-rapida
    """
    def __init__(self, num_classes=1, dropout_rate=0.2):
        super(UAVShipNet, self).__init__()
        
        # Bloques de extraccion jerarquica de caracteristicas (sin asfixiar capas tempranas)
        self.block1 = ConvBlock(in_channels=3, out_channels=32, dropout_p=0.0)   # -> 40x40
        self.block2 = ConvBlock(in_channels=32, out_channels=64, dropout_p=0.0)   # -> 20x20
        self.block3 = ConvBlock(in_channels=64, out_channels=128, dropout_p=0.10) # -> 10x10
        
        # Bloque profundo de contexto
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling (GAP)
        )
        
        # Cabeza de clasificacion mecatronica (Baja latencia)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes) # Logits de salida
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x):
        """Retorna la probabilidad calibrada P(Barco | X) en rango [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

def count_parameters(model):
    """Calcula el numero total de parametros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = UAVShipNet()
    total_params = count_parameters(model)
    print(f"Arquitectura 'UAVShipNet' instanciada exitosamente.")
    print(f"Total parametros entrenables: {total_params:,}")
    
    # Prueba de paso hacia adelante con tensor simulado (Batch de 4 imagenes 80x80)
    dummy_input = torch.randn(4, 3, 80, 80)
    output = model(dummy_input)
    probs = model.predict_proba(dummy_input)
    print(f"Forma de entrada:  {dummy_input.shape}")
    print(f"Forma de salida:   {output.shape}")
    print(f"Probabilidades de salida: {probs.squeeze().tolist()}")
