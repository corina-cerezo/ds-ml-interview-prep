# Entendiendo el test de Kolmogorov-Smirnov

*Notas teóricas sobre el fundamento matemático de la prueba.*



## Introducción

El test de Kolmogorov-Smirnov (KS) es una prueba no paramétrica para comparar distribuciones. En su versión más común, contrasta la distribución de una muestra contra una distribución teórica especificada; en su versión de dos muestras, contrasta dos muestras independientes entre sí. Aunque popularmente se le asocia con pruebas de normalidad, esa asociación es un accidente del uso: la prueba funciona para cualquier distribución continua especificada bajo la hipótesis nula, no solo la normal.

Lo que hace al KS particularmente elegante es la teoría que lo sustenta. Detrás de un estadístico aparentemente simple (la máxima diferencia entre dos funciones de distribución) hay tres pilares teóricos que se entrelazan: 
* El teorema de Glivenko-Cantelli (que usa la Ley Fuerte de los Grandes Números en una parte de su demostración), 
* la propiedad libre de distribución y 
* la convergencia al puente browniano. 

Este documento desarrolla cada uno con suficiente detalle para entender por qué la prueba funciona, no solo cómo aplicarla.



## 1. El problema y la función de distribución empírica

Sea $X_1, X_2, \dots, X_n$ una muestra aleatoria iid de una distribución con función de distribución acumulada $F$, desconocida. El problema fundamental de la estadística no paramétrica es: ¿qué se puede decir de $F$ a partir de la muestra?

La respuesta natural es construir un estimador de $F$ a partir de los datos. Ese estimador es la **función de distribución empírica** (FDE):

> **Definición (Función de distribución empírica).** Dada una muestra $X_1, \dots, X_n$, la función de distribución empírica $F_n$ se define como
> $$F_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(X_i \leq x), \quad x \in \mathbb{R}$$
> donde $\mathbf{1}(\cdot)$ es la función indicadora.

Es decir, $F_n(x)$ es la proporción de observaciones de la muestra que son menores o iguales a $x$. Como función de $x$, es una función escalonada que sube en cada observación y permanece constante entre ellas. Si todas las observaciones son distintas, cada salto mide exactamente $1/n$. Si hay $k$ observaciones empatadas en un mismo valor, el salto en ese punto mide $k/n$, porque cada observación contribuye con $1/n$ y las contribuciones se acumulan en el mismo punto.

La FDE es **continua por la derecha**, no continua en sentido estricto. La razón está en el "$\leq$" de la indicadora: cuando $x$ es exactamente una observación, esa observación ya está contada, por lo que $F_n(x)$ toma el valor del escalón superior. Si la definición usara desigualdad estricta, sería continua por la izquierda.

> **Ilustración 1:** muestra pequeña con su FDE escalonada, con la posibilidad de ver cómo cambia al aumentar $n$ y cómo, en muestras con valores repetidos, aparecen saltos de tamaño $k/n$.
>
> [![FDE escalonada](https://raw.githubusercontent.com/corina-cerezo/ds-ml-interview-prep/main/04_deep_dives/kolmogorov_smirnov/images/01_empirical_cdf_es.png)](https://htmlpreview.github.io/?https://github.com/corina-cerezo/ds-ml-interview-prep/blob/main/04_deep_dives/kolmogorov_smirnov/interactive/01_empirical_cdf_es.html)
>
> *¡Da clic a la imagen para ir a la ilustración interactiva! Observa cómo cuando $n$ crece la FDE converge a la distribución teórica.*


### 1.1 Propiedades clave de la FDE

La FDE es un estimador notablemente bueno de $F$, con las siguientes propiedades:

**1. Insesgada puntualmente.** Para cada $x$ fijo:

$$
\begin{aligned}
\mathbb{E}[F_n(x)] &= \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(X_i \leq x)\right] \\
&= \frac{1}{n}\sum_{i=1}^{n} \mathbb{E}[\mathbf{1}(X_i \leq x)] \\
&= \frac{1}{n}\sum_{i=1}^{n} P(X_i \leq x) \\
&= \frac{1}{n} \cdot n \cdot F(x) \\
&= F(x)
\end{aligned}
$$

Los pasos clave: en la segunda igualdad se usa la linealidad de la esperanza; en la tercera, que la esperanza de una indicadora es la probabilidad del evento ($\mathbb{E}[\mathbf{1}(A)] = P(A)$); en la cuarta, que $P(X_i \leq x) = F(x)$ por definición de la función de distribución, y como todas las $X_i$ son idénticamente distribuidas, la suma de $n$ términos iguales da $n \cdot F(x)$.


**2. Varianza conocida.** Para cada $x$ fijo:

$$
\begin{aligned}
\text{Var}(F_n(x)) &= \text{Var}\left(\frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(X_i \leq x)\right) \\
&= \frac{1}{n^2} \sum_{i=1}^{n} \text{Var}(\mathbf{1}(X_i \leq x)) \\
&= \frac{1}{n^2} \cdot n \cdot F(x)(1 - F(x)) \\
&= \frac{F(x)(1 - F(x))}{n}
\end{aligned}
$$

Los pasos clave: en la segunda igualdad se usa que la varianza de una suma de variables **independientes** es la suma de las varianzas (las $X_i$ son iid, por lo que las indicadoras $\mathbf{1}(X_i \leq x)$ también lo son), y la constante $1/n$ sale al cuadrado por la propiedad $\text{Var}(aY) = a^2 \text{Var}(Y)$. En la tercera, se usa que cada $\mathbf{1}(X_i \leq x)$ es una variable Bernoulli con probabilidad de éxito $F(x)$, cuya varianza es $p(1-p) = F(x)(1-F(x))$. La suma de $n$ términos iguales da $n \cdot F(x)(1-F(x))$.

Una consecuencia inmediata es que $\text{Var}(F_n(x)) \to 0$ cuando $n \to \infty$, lo que implica convergencia en media cuadrática y, por la desigualdad de Chebyshev, convergencia en probabilidad a $F(x)$.

**3. Consistencia puntual (casi segura).** Aplicando la Ley Fuerte de los Grandes Números a las indicadoras $\mathbf{1}(X_i \leq x)$, que son iid Bernoulli con esperanza $F(x)$, se obtiene un resultado más fuerte que el de la propiedad 2:

$$F_n(x) \xrightarrow{\text{c.s.}} F(x) \quad \Longleftrightarrow \quad P\left(\lim_{n \to \infty}|F_n(x) - F(x)| = 0 \right) = 1$$

Es decir, la convergencia no solo ocurre en probabilidad, sino casi seguramente: para casi toda realización del experimento, la sucesión $F_n(x)$ se estabiliza efectivamente en $F(x)$.

Conviene notar que este resultado es **puntual**: garantiza convergencia para cada $x$ fijo, pero no dice nada sobre el comportamiento simultáneo en todos los puntos. Para la prueba KS necesitaremos algo más fuerte: convergencia **uniforme**, que se obtiene en la sección 3 con el teorema de Glivenko-Cantelli.


**En conclusión,** estas propiedades juntas hacen de $F_n$ un buen estimador de $F$: 
* **insesgada** (en promedio acierta el valor verdadero), 
* **con varianza decreciente** (la dispersión alrededor de ese valor se reduce conforme crece la muestra), y 
* **consistente casi seguramente** (cada trayectoria muestral converge efectivamente al valor verdadero).

En conjunto, esto significa que $F_n(x)$ no solo es correcto en expectativa, sino que con suficientes datos se acerca arbitrariamente al valor real con probabilidad uno.

Un punto conceptual importante: la FDE es un **estimador funcional**, no escalar. Mientras que la mayoría de los estimadores estiman un parámetro real (la media, la varianza, un percentil), la FDE estima un objeto en un espacio de dimensión infinita: la función completa $F$. Esto la coloca en una categoría distinta y abre la puerta a toda la estadística no paramétrica moderna.

> **Nota.** $F_n$ tiene además otras propiedades notables que no se desarrollan aquí. Entre ellas, es el estimador de máxima verosimilitud no paramétrico de $F$: entre todos los posibles estimadores de $F$ sin suponer una forma paramétrica, $F_n$ es el que maximiza la verosimilitud de los datos observados. 


### 1.2 Una nota sobre empates y el supuesto de continuidad

Si $F$ es continua, la probabilidad de que dos observaciones coincidan es cero. La razón formal es que, para $F$ continua, $P(X = a) = F(a) - F(a^-) = 0$ para todo $a$, y de ahí, por independencia,
$$P(X_i = X_j) = \int P(X_i = y)\, dF_{X_j}(y) = 0,$$
porque $P(X_i=y)=0$ para todo $y$.

Geométricamente, el evento $\{X_i = X_j\}$ corresponde a la recta diagonal en $\mathbb{R}^2$, que tiene medida de Lebesgue cero. Esto importa para el KS porque toda la teoría que viene supone $F_0$ (la distribución hipotetizada) continua. Cuando los datos presentan empates por discreción o redondeo, los supuestos se violan parcialmente y la distribución exacta del estadístico cambia.



## 2. El estadístico $D_n$ de Kolmogorov

Si $F_n$ es un buen estimador de $F$, y se quiere contrastar la hipótesis de que los datos provienen de una distribución específica $F_0$, parece natural medir la discrepancia entre $F_n$ y $F_0$. Kolmogorov (1933) propuso medirla con el supremo de la diferencia absoluta:

> **Definición (Estadístico de Kolmogorov).** Dada una muestra $X_1, \dots, X_n$ y una función de distribución $F_0$ especificada, el estadístico de Kolmogorov es
> $$D_n = \sup_{x \in \mathbb{R}} |F_n(x) - F_0(x)|$$

Geométricamente, $D_n$ es la mayor separación vertical entre la FDE (escalonada) y la función de distribución acumulada (FDA) hipotetizada (suave, si $F_0$ es continua). Esta forma de medir discrepancia se conoce como **norma del supremo** o $L^\infty$ entre las funciones $F_n$ y $F_0$: en lugar de promediar las diferencias sobre todos los puntos, se toma el peor desfase. 

Las hipótesis de la prueba son:
$$H_0: F = F_0 \quad \text{vs} \quad H_1: F \neq F_0$$

> **Ilustración 2:** FDE y FDA teórica superpuestas, con la barra vertical que representa $D_n$ resaltada. Permite ver cómo cambia $D_n$ al modificar la muestra.
>
> [![Estadístico de Kolmogorov](https://raw.githubusercontent.com/corina-cerezo/ds-ml-interview-prep/main/04_deep_dives/kolmogorov_smirnov/images/02_ks_statistic_es.png)](https://htmlpreview.github.io/?https://github.com/corina-cerezo/ds-ml-interview-prep/blob/main/04_deep_dives/kolmogorov_smirnov/interactive/02_ks_statistic_es.html)
>
> *¡Da clic a la imagen para ir a la ilustración interactiva! Observa cómo $D_n$ siempre marca la mayor diferencia, a veces queda por encima y a veces por debajo de la distribución teórica.*

### ¿Por qué el supremo?

Hay otras formas razonables de cuantificar la discrepancia entre dos distribuciones. Las más conocidas son:

- **Kolmogorov-Smirnov:** $D_n = \sup_x |F_n(x) - F_0(x)|$. Distancia $L^\infty$.
- **Cramér-von Mises:** $W_n^2 = n \int (F_n(x) - F_0(x))^2\, dF_0(x)$. Distancia $L^2$ ponderada.
- **Anderson-Darling:** $A_n^2 = n \int \frac{(F_n(x) - F_0(x))^2}{F_0(x)(1-F_0(x))}\, dF_0(x)$. Cramér-von Mises con peso que enfatiza las colas.


Las tres son pruebas de bondad de ajuste basadas en la FDE, pero con sensibilidades distintas. El KS tiende a ser más sensible a diferencias localizadas cerca del centro de la distribución, porque bajo \(H_0\) la varianza puntual de \(F_n(x) - F_0(x)\) es \(F_0(x)(1-F_0(x))/n\), la cual es máxima cuando \(F_0(x)=0.5\). Anderson-Darling detecta mejor diferencias en las colas, gracias al peso \(1/[F_0(1-F_0)]\) que crece en los extremos.


Para que un estadístico de discrepancia sea útil como prueba, se necesitan dos propiedades: que sea pequeño cuando $H_0$ es cierta (para no rechazar erróneamente), y que su distribución bajo $H_0$ sea conocida (para fijar valores críticos). Los siguientes tres pilares dan exactamente eso.



## 3. Glivenko-Cantelli: la justificación de la prueba

El primer pilar de la prueba KS es el teorema que garantiza que $D_n$ sea pequeño bajo $H_0$.

> **Teorema (Glivenko-Cantelli, 1933).** Sea $X_1, X_2, \dots$ una sucesión iid con función de distribución $F$, y sea $F_n(x)$ la FDE. Entonces
> $$\sup_{x \in \mathbb{R}} |F_n(x) - F(x)| \xrightarrow{\text{c.s.}} 0 \quad \text{cuando } n \to \infty.$$
> O de forma equivalente
> $$P\left(\lim_{n \to \infty} \sup_{x \in \mathbb{R}} |F_n(x) - F(x)| = 0\right) = 1.$$

En palabras, este teorema nos da dos propiedades importantes de la FDE: converge **uniformemente** y **casi seguramente** a la verdadera FDA. Aunque no pongo la demostración, estas dos cualidades de la convergencia son interesantes y vale la pena discutirlas por separado.

### Convergencia uniforme

La convergencia uniforme es más fuerte que la convergencia puntual: dice que el peor error sobre **toda** la recta real tiende a cero. La Ley Fuerte de los Grandes Números, aplicada a las indicadoras, ya da convergencia puntual ($F_n(x) \to F(x)$ para cada $x$ fijo). Pero la convergencia puntual de una sucesión de funciones no implica, en general, convergencia uniforme: la velocidad de convergencia podría ser distinta en diferentes puntos, y el supremo no necesariamente convergería. Lo no trivial de Glivenko-Cantelli es precisamente promover la convergencia puntual a uniforme, mediante un argumento de partición de la recta con cuantiles de $F$ y control simultáneo en cada celda.

### Convergencia casi segura

La convergencia casi segura es más fuerte que la convergencia en probabilidad. Mientras que la convergencia en probabilidad dice que "para cada $n$ grande, es improbable estar lejos del límite" (la sucesión puede oscilar mientras la probabilidad de estar lejos disminuya), la convergencia casi segura dice que "para casi toda realización del experimento, la sucesión efectivamente se estabiliza en el límite". Es decir, trayectoria por trayectoria.

> **Ilustración 3:** se muestra una sucesión de FDEs para tamaños crecientes de muestra y cómo el peor error $D_n$ disminuye conforme $n$ crece.
>
> [![Convergencia de Glivenko-Cantelli](https://raw.githubusercontent.com/corina-cerezo/ds-ml-interview-prep/main/04_deep_dives/kolmogorov_smirnov/images/03_glivenko_cantelli_es.png)](https://htmlpreview.github.io/?https://github.com/corina-cerezo/ds-ml-interview-prep/blob/main/04_deep_dives/kolmogorov_smirnov/interactive/03_glivenko_cantelli_es.html)
>
> *¡Da clic a la imagen para ir a la ilustración interactiva! Observa cómo cuando $n$ crece $D_n$ se hace más pequeño.*

### Por qué importa para el KS

El resultado se traduce directamente en el contexto del test. Si $H_0$ es cierta, los datos provienen efectivamente de $F_0$, es decir $F = F_0$, y entonces $D_n = \sup_x|F_n - F_0| \to 0$ casi seguramente. Por el contrario, si $H_0$ es falsa, los datos provienen de alguna $F \neq F_0$, y aplicando Glivenko-Cantelli a la verdadera $F$ se obtiene $F_n \to F$ uniformemente. Entonces, por la desigualdad triangular,
$$D_n = \sup_x|F_n - F_0| \xrightarrow{\text{c.s.}} \sup_x|F - F_0| > 0$$

Es decir, bajo $H_1$, el estadístico tiende a una constante estrictamente positiva. Esto implica que la potencia de la prueba tiende a 1 cuando $n \to \infty$, lo que en términos formales se enuncia como:

> **Consecuencia:** la prueba KS es **consistente** contra cualquier alternativa fija $F \neq F_0$.

Esta asimetría entre el comportamiento bajo $H_0$ y bajo $H_1$ es lo que justifica la regla de decisión: rechazar $H_0$ cuando $D_n$ es grande.

### Por qué se le conoce como el "teorema fundamental de la estadística"

El apodo es conceptual, no técnico. Glivenko-Cantelli formaliza la idea filosófica más profunda de toda la inferencia estadística: con suficientes datos, la muestra revela la distribución subyacente. Sin un resultado de este tipo, no habría justificación rigurosa para hacer inferencia a partir de muestras finitas. La estadística paramétrica clásica (pruebas $t$, $\chi^2$, ANOVA) suele apoyarse en la Ley de los Grandes Números y el Teorema Central del Límite, no en Glivenko-Cantelli explícitamente; pero la legitimidad última de cualquier inferencia descansa en el principio que este teorema captura.

Por estas razones, el teorema a veces se conoce como el "teorema fundamental de la estadística", apodo que aunque no es universal, refleja su papel central en justificar la inferencia estadística como empresa.



## 4. La propiedad libre de distribución

El segundo pilar de la prueba KS responde a la pregunta: ¿la prueba se comporta igual cuando contrastamos contra una normal, una exponencial, una gamma, o cualquier otra distribución continua? Sin esta propiedad, cada paquete estadístico tendría que implementar una rutina distinta para cada posible $F_0$.

> **Teorema (Propiedad libre de distribución).** Sea $F_0$ una función de distribución continua y $X_1, \dots, X_n$ una muestra iid de $F_0$. Entonces la distribución del estadístico
> $$D_n = \sup_{x \in \mathbb{R}} |F_n(x) - F_0(x)|$$
> es la misma para toda $F_0$ continua.

En otras palabras, la distribución de $D_n$ no depende de $F_0$. No importa si en la prueba KS estamos contrastando nuestra muestra contra una normal, una gamma, una exponencial, etc.: la distribución del estadístico de prueba es siempre la misma.

Si se analiza con calma, ¡esto es increíble! La propiedad libre de distribución no nos dice **cuál** es la distribución de $D_n$, solo nos dice que es **única**, sin importar contra qué $F_0$ estemos contrastando. Es decir, la distribución del máximo error entre $F_n$ y $F_0$ tiene una forma propia, independiente de la $F_0$ específica que aparezca en cada problema. 

Para la demostración se requiere conocer el siguiente lema.

### La transformación integral de probabilidad

> **Lema (Transformación integral de probabilidad).** Si $X$ es una variable aleatoria con función de distribución $F$ continua, entonces
> $$U := F(X) \sim \text{Uniforme}(0,1)$$

**Demostración.** Supóngase primero que $F$ es estrictamente creciente, de modo que existe su inversa $F^{-1}$. Para $u \in [0,1]$:
$$P(U \leq u) = P(F(X) \leq u) = P(X \leq F^{-1}(u)) = F(F^{-1}(u)) = u$$

Es decir, $P(U \leq u) = u$, que es la FDA de la distribución Uniforme(0,1). Cuando $F$ no es estrictamente creciente (puede tener mesetas), se reemplaza $F^{-1}$ por la función cuantil generalizada $F^{-1}(u) = \inf\{x: F(x) \geq u\}$ y el argumento sigue siendo válido. $\blacksquare$

La intuición geométrica del resultado es la siguiente: la función $F$ "endereza" la distribución. Donde la densidad $f = F'$ es alta, $F$ sube rápidamente, y muchas observaciones se proyectan sobre un tramo amplio del eje vertical. Donde la densidad es baja, $F$ sube lentamente, y las pocas observaciones que caen ahí se proyectan sobre un tramo estrecho. Las dos no uniformidades se cancelan exactamente, y el resultado es que las alturas $F(X)$ se distribuyen uniformemente en $[0,1]$.

> **Ilustración 4:** distribución original $X \sim F$ y su transformación $F(X)$. Permite seleccionar varias distribuciones (normal, exponencial, beta) y observar cómo, sin importar la forma original, las alturas siempre se distribuyen uniformemente.
>
> [![Transformación integral de probabilidad](https://raw.githubusercontent.com/corina-cerezo/ds-ml-interview-prep/main/04_deep_dives/kolmogorov_smirnov/images/04_probability_integral_transform_es.png)](https://htmlpreview.github.io/?https://github.com/corina-cerezo/ds-ml-interview-prep/blob/main/04_deep_dives/kolmogorov_smirnov/interactive/04_probability_integral_transform_es.html)
>
> *¡Da clic a la imagen para ir a la ilustración interactiva! Observa cómo sin importar la distribución original, las alturas se distribuyen uniformemente.*

### Demostración del teorema

Sea $X_1, \dots, X_n$ iid con FDA $F_0$ continua. Defínase $U_i := F_0(X_i)$. Por la transformación integral, cada $U_i$ es Uniforme(0,1), y como las $X_i$ son independientes y $F_0$ es una función determinista, las $U_i$ son iid Uniforme(0,1).

Aplíquese el cambio de variable $y = F_0(x)$ en el supremo. Como $F_0$ es continua y (asúmase por simplicidad) estrictamente creciente, $x$ recorre $\mathbb{R}$ si y solo si $y$ recorre $(0,1)$. Entonces:
$$D_n = \sup_{x \in \mathbb{R}}|F_n(x) - F_0(x)| = \sup_{y \in (0,1)}|F_n(F_0^{-1}(y)) - y|$$

donde se usó $F_0(F_0^{-1}(y)) = y$. Ahora obsérvese que:
$$F_n(F_0^{-1}(y)) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(X_i \leq F_0^{-1}(y)) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(F_0(X_i) \leq y) = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(U_i \leq y) =: G_n(y)$$

donde $G_n$ es la FDE de la muestra uniforme $U_1, \dots, U_n$. Sustituyendo:
$$D_n = \sup_{y \in (0,1)}|G_n(y) - y|$$

En esta forma, $F_0$ ha desaparecido completamente: $D_n$ tiene exactamente la misma distribución que el estadístico KS aplicado a una muestra Uniforme(0,1) contra su propia FDA teórica. Como esto no depende de cuál era la $F_0$ original, la distribución de $D_n$ es universal entre todas las $F_0$ continuas. $\blacksquare$

### Consecuencias

Esta propiedad tiene tres implicaciones importantes:

1. **La prueba KS se implementa de forma universal.** Cualquier paquete estadístico puede aplicar el test contra cualquier $F_0$ continua sin necesidad de rutinas específicas por distribución, porque la distribución de referencia bajo $H_0$ es siempre la misma.
2. **Las simulaciones se hacen una sola vez.** Para obtener cuantiles empíricos del estadístico bajo $H_0$, basta simular muestras uniformes; no se necesita simular para cada $F_0$ posible.
3. **El estudio de la distribución asintótica se reduce al caso uniforme.** Para encontrar la distribución límite de $\sqrt{n}\,D_n$, basta analizar el caso de una muestra Uniforme(0,1).

Esta última consecuencia es la que abre la puerta al tercer pilar.



## 5. La distribución asintótica de $\sqrt{n}\,D_n$

Hasta este punto se sabe que $D_n \to 0$ bajo $H_0$ y que su distribución no depende de $F_0$. Falta saber cuál es esa distribución concreta para poder calcular p-valores y valores críticos.

La respuesta fue dada por Kolmogorov en su artículo original de 1933:

> **Teorema (Kolmogorov, 1933).** Si $F_0$ es continua y $H_0$ es cierta,
> $$\lim_{n \to \infty} P\left(\sqrt{n}\,D_n \leq x\right) = K(x) = 1 - 2\sum_{k=1}^{\infty}(-1)^{k-1}e^{-2k^2 x^2}, \quad x > 0$$
> La función $K$ se conoce como la **distribución de Kolmogorov**.

Es importante leer con cuidado lo que dice este teorema. **No** afirma que $D_n$ tenga distribución $K$; afirma que $\sqrt{n}\,D_n$ converge en distribución a $K$ cuando $n \to \infty$. El factor $\sqrt{n}$ es el escalamiento correcto: como la varianza de $F_n(x)$ es del orden $1/n$, su desviación estándar es del orden $1/\sqrt{n}$, y multiplicar por $\sqrt{n}$ estabiliza la escala.

Para $n$ finito existe también una distribución exacta de $D_n$, calculable mediante recursiones combinatorias, pero no tiene una forma cerrada simple. En la práctica, los paquetes estadísticos usan la distribución exacta para $n$ pequeño y la aproximación asintótica $K$ para $n$ moderado o grande (típicamente $n \geq 30$).

La serie de $K$ converge muy rápidamente (los términos decaen como gaussianas), por lo que en la práctica basta sumar unos pocos términos para obtener precisión numérica suficiente.

> **Ilustración 5:** histograma simulado de $\sqrt{n}\,D_n$ bajo $H_0$ para distintos valores de $n$, superpuesto con la densidad teórica de $K$. Permite ver cómo la distribución empírica converge a la teórica al crecer $n$.
>
> [![Distribución de Kolmogorov](https://raw.githubusercontent.com/corina-cerezo/ds-ml-interview-prep/main/04_deep_dives/kolmogorov_smirnov/images/05_kolmogorov_distribution_es.png)](https://htmlpreview.github.io/?https://github.com/corina-cerezo/ds-ml-interview-prep/blob/main/04_deep_dives/kolmogorov_smirnov/interactive/05_kolmogorov_distribution_es.html)
>
> *¡Da clic a la imagen para ir a la ilustración interactiva! Observa cómo al crecer $n$, el histograma simulado se acerca a la densidad teórica de $K$.*

### Una conexión sorprendente: el puente browniano 

La forma en que Kolmogorov originalmente derivó esta distribución fue combinatoria, pero la perspectiva moderna reveló una conexión con los procesos estocásticos. 🤯

De alguna forma (que probablemente entendería si supiera procesos estocásticos) la diferencia entre $F_n$ y $F_0$ (escalada por $\sqrt{n}$), se comporta como una caminata aleatoria continua llamada **puente browniano**. La distribución de Kolmogorov $K$ resulta ser exactamente la distribución del supremo del valor absoluto de ese puente browniano.

No profundicé en los fundamentos técnicos de esta conexión, pero me pareció interesante mencionarlo: una prueba estadística sobre funciones de distribución empíricas termina conectándose con un objeto central de la teoría de la probabilidad y procesos estocásticos. 



## 6. El caso de dos muestras (donde Smirnov contribuyó)

Hasta ahora se ha considerado el caso de comparar una muestra contra una distribución teórica. La versión de dos muestras del KS, debida a Smirnov (1939), contrasta dos muestras independientes entre sí.

Sean $X_1, \dots, X_n$ iid con FDA $F$, y $Y_1, \dots, Y_m$ iid con FDA $G$, independientes entre sí. Sean $F_n$ y $G_m$ las respectivas FDEs. El estadístico es:

> **Definición (Estadístico KS de dos muestras).**
> $$D_{n,m} = \sup_{x \in \mathbb{R}} |F_n(x) - G_m(x)|$$

Las hipótesis son $H_0: F = G$ (sin especificar cuál) vs $H_1: F \neq G$.


> **Ilustración 6:** dos FDEs superpuestas con la máxima diferencia resaltada. Permite comparar dos muestras de distintos tamaños.
>
> [![KS de dos muestras](https://raw.githubusercontent.com/corina-cerezo/ds-ml-interview-prep/main/04_deep_dives/kolmogorov_smirnov/images/06_two_samples_es.png)](https://htmlpreview.github.io/?https://github.com/corina-cerezo/ds-ml-interview-prep/blob/main/04_deep_dives/kolmogorov_smirnov/interactive/06_two_samples_es.html)
>
> *¡Da clic a la imagen para ir a la ilustración interactiva! Observa cómo $D_{n,m}$ marca la mayor separación vertical entre las dos FDEs.*


La teoría es análoga al caso de una muestra, con una diferencia importante: el escalamiento.

> **Teorema (Smirnov, 1939).** Bajo $H_0$ con $F = G$ continua,
> $$\lim_{n, m \to \infty} P\left(\sqrt{\frac{nm}{n+m}}\,D_{n,m} \leq x\right) = K(x)$$
> donde $K$ es la misma distribución de Kolmogorov del caso de una muestra.

El factor $\sqrt{nm/(n+m)}$ es el "tamaño efectivo de muestra" cuando hay dos grupos. Cuando $n = m$, se reduce a $\sqrt{n/2}$. Cuando $m \to \infty$ con $n$ fijo, tiende a $\sqrt{n}$, recuperando el caso de una muestra (lo cual tiene sentido: una muestra infinita actúa como si fuera la verdadera distribución teórica).

La propiedad libre de distribución también se mantiene: bajo $H_0$, la distribución de $D_{n,m}$ no depende de cuál sea la $F$ común, siempre que sea continua. 

Una ventaja práctica importante del KS de dos muestras es que **no requiere conocer la distribución subyacente**. Es completamente no paramétrico, lo que lo hace una herramienta natural para comparar dos grupos sin asumir normalidad u otra forma específica.



## 7. Limitaciones y guía práctica

Aunque el KS es una prueba teóricamente elegante, conviene tener presentes sus limitaciones antes de usarla.

**Empates.** La teoría asume $F_0$ continua, lo que implica que con probabilidad uno no hay observaciones repetidas. Cuando los datos presentan empates (por discreción de la variable, por redondeo, o por escalas con pocos decimales), la distribución del estadístico bajo $H_0$ deja de ser la teórica y se vuelve **conservadora**: la prueba rechaza menos de lo nominal. Paquetes como R emiten una advertencia en este caso. Existen extensiones del test para distribuciones discretas, pero no están implementadas de forma nativa en la mayoría del software estándar; en la práctica suele preferirse un test de permutaciones para calcular el p-valor exacto, o cambiar a una prueba como Cramér-von Mises o chi-cuadrado.

**Parámetros estimados.** Si los parámetros de $F_0$ se estiman a partir de la misma muestra (por ejemplo, usando $\bar{x}$ y $s^2$ para construir una normal hipotetizada), la propiedad libre de distribución **no se preserva** y los valores críticos de la distribución de Kolmogorov ya no son los correctos. La prueba se vuelve conservadora y la probabilidad real de rechazar bajo $H_0$ queda por debajo del nivel nominal $\alpha$. La **corrección de Lilliefors** (1967) recalcula los valores críticos vía simulación para el caso específico de normalidad con parámetros estimados; existen correcciones análogas para otras distribuciones.

**Sensibilidad en colas.** El KS es más sensible a diferencias cerca del centro de la distribución que en las colas, porque la FDE varía más rápido cerca de la mediana. En aplicaciones donde las colas son lo que interesa (riesgo financiero, valores extremos), la prueba de Anderson-Darling, que pondera por $1/[F_0(1-F_0)]$, suele ser más adecuada.

**Comparación con otras pruebas.** Para pruebas específicas de normalidad, **Shapiro-Wilk** suele tener mayor potencia que el KS (incluso con la corrección de Lilliefors). Para bondad de ajuste general, **Anderson-Darling** y **Cramér-von Mises** son alternativas competitivas con sensibilidades distintas.

**Tamaño de muestra y significancia.** Como toda prueba consistente, con $n$ suficientemente grande el KS detecta diferencias arbitrariamente pequeñas como estadísticamente significativas. Esto no es una falla de la prueba, sino una característica de la inferencia frecuentista: la significancia estadística no equivale a significancia práctica. Para muestras muy grandes, conviene complementar el p-valor con una medida del tamaño de la discrepancia (por ejemplo, el propio valor de $D_n$).



## Referencias

### Bibliografía revisada para la elaboración de este documento

1. Khoshnevisan, D. *Empirical Processes, and the Kolmogorov–Smirnov Statistic.* Notas del curso, Universidad de Utah. Disponible en: [https://www.math.utah.edu/~davar/ps-pdf-files/Kolmogorov-Smirnov.pdf](https://www.math.utah.edu/~davar/ps-pdf-files/Kolmogorov-Smirnov.pdf)

2. Gibbons, J. D. y Chakraborti, S. *Nonparametric Statistical Inference.* Disponible en: [https://sadbhavnapublications.org/research-enrichment-material/2-Statistical-Books/Nonparametric-Statistics-Theory-and-Methods.pdf](https://sadbhavnapublications.org/research-enrichment-material/2-Statistical-Books/Nonparametric-Statistics-Theory-and-Methods.pdf)


### Artículos originales

- Kolmogorov, A. N. (1933). *Sulla determinazione empirica di una legge di distribuzione.* Giornale dell'Istituto Italiano degli Attuari, 4, 83–91.
- Smirnov, N. V. (1939). *Sur les écarts de la courbe de distribution empirique.* Matematicheskii Sbornik, 6, 3–26.
- Lilliefors, H. (1967). *On the Kolmogorov-Smirnov test for normality with mean and variance unknown.* Journal of the American Statistical Association, 62, 399–402.

---

*La idea de elaborar este documento me surgió a partir de conversaciones con Claude (Anthropic), y lo usé para la revisión de redacción y la creación de las ilustraciones interactivas.*
