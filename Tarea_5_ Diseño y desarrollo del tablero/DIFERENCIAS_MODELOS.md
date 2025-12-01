# Diferencias entre Modelos de Clasificación

## 📊 MODELO TEC vs UNIANDES

### **MODELO TEC: Clasificación por Composite Score**

**Criterio:** Combina 3 factores con pesos iguales

```
Composite Score = (Rating Normalizado) - (Precio Normalizado) + (Amenities Normalizado)
                    ↑ Positivo              ↑ Negativo          ↑ Positivo
```

**Interpretación:**
- ✅ Rating ALTO = propiedad buena
- ✅ Precio BAJO = buena oferta  
- ✅ Amenities ALTO = más servicios
- 🎯 **RECOMENDABLE si:** Composite Score ≥ mediana del dataset

**Features usados:** 6
- price
- accommodates
- bedrooms
- bathrooms
- amenities_number
- review_scores_rating

**Algoritmo:** Regresión Logística

**Ejemplo:**
- Precio $150 (bajo), Rating 4.8/5 (alto), 15 amenities (alto) = RECOMENDADO
- Precio $400 (alto), Rating 4.9/5 (alto), 5 amenities (bajo) = NO RECOMENDADO

---

### **MODELO UNIANDES: Clasificación Multicriterio Estricto**

**Criterio:** Todos estos deben cumplirse simultáneamente (AND):

```
RECOMENDABLE si:
  ✓ Precio ≤ $200  (presupuesto limitado)
  ✓ Rating ≥ 4.5   (calidad mínima)
  ✓ Bedrooms ≥ 1   (tiene espacios)
  ✓ Amenities ≥ 5  (servicios básicos)
  ✓ Host Response Rate ≥ 0.79 (anfitrión responsable)
```

**Interpretación:**
- Busca propiedades **económicas Y de calidad**
- Requiere buena comunicación del anfitrión
- Usa todos los datos disponibles en el dataset (después de one-hot encoding)

**Features usados:** ~100+ (después de get_dummies con todas las variables)

**Algoritmo:** Red Neuronal (64 → 32 → 1 neuronas)

**Ejemplo:**
- Precio $180, Rating 4.6, 2 bedrooms, 8 amenities, host_response 0.85 = RECOMENDADO
- Precio $220, Rating 4.9, 3 bedrooms, 10 amenities, host_response 0.90 = NO RECOMENDADO (precio alto)

---

## 🔍 RESUMEN COMPARATIVO

| Aspecto | TEC | UNIANDES |
|---------|-----|----------|
| **Focus** | Mejor relación precio-calidad | Opciones económicas de calidad |
| **Precio Ideal** | Variable (balanceado) | ≤ $200 (presupuesto limitado) |
| **Features** | 6 simples | ~100+ (incluye categorías) |
| **Algoritmo** | Logistic Regression | Red Neuronal |
| **Criterio** | Composite Score | Multicriterio estricto |
| **Recomendación** | Si composite_score > mediana | Si TODOS los requisitos se cumplen |

---

## 💡 CUÁNDO USAR CADA UNO

**Usa TEC Clasificación cuando:**
- Quieres encontrar el mejor valor por dinero
- La importancia es la relación precio-calidad
- Aceptas pagar más si obtienes muchas amenities

**Usa Uniandes Clasificación cuando:**
- Tienes presupuesto limitado (<$200)
- Buscas garantía de comunicación del anfitrión
- Necesitas espacios separados (bedrooms)
- Requieres servicios básicos (≥5 amenities)
