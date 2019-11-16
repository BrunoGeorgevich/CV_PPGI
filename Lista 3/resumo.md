<center><h1>Calibração de Câmera</h1></center>

​	No momento do cálculo da calibração de uma câmera o OpenCV leva em conta dois fatores importantes, os radiais e os  tangenciais. Os fatores radiais são manifestados nas famosas lentes 'olho de peixe', as quais deformam radialmente a imagem correlação ao centro. Os tangenciais ocorrem pelo fato das lentes não serem perfeitamente paralelas ao plano da imagem a ser capturada, apresentando uma distorção de captura na imagem. A imagem da esquerda ilustra um exemplo de distorção radial, enquanto que a da direita de distorção tangencial.  As distorções radial e tangencial são descritas pelas equação abaixo, respectivamente. Sendo o ponto (x,y)  pertencente a imagem não distorcida e o ($x_{distorced}$,$y_{distorced}$) a imagem distorcida.

​     $\left\{\begin{matrix}
x_{distorced} = x(1 + k_{1}r² + k_{2}r⁴ + k_{3}r⁶)\\ 
y_{distorced} = y(1 + k_{1}r² + k_{2}r⁴ + k_{3}r⁶)
\end{matrix}\right .$                $\left\{\begin{matrix}
x_{distorced} = x + [2p_1xy + p2(r² + 2x²)]\\ 
y_{distorced} = y + [2p_2xy + p1(r² + 2y²)]
\end{matrix}\right .$



<center>
	<div>
    	<img src="http://s5.favim.com/orig/54/eye-fish-fish-eye-mar-oceano-Favim.com-523288.jpg" alt="" style="height: 150px" /> 
	    <img src="https://live.staticflickr.com/2695/4114302348_89281119ef_b.jpg" style="height: 150px"  />
    </div>
</center>

​	Desta forma, devido as distorções previamente apresentadas, são obtidas cinco variáveis de distorção, também chamadas de **coeficientes de distorção**.  Descritas da seguinte forma:
$$
distortion\_coefficients = (k_1 \; k_2 \; p_1 \; p_2 \; k_3)
$$
​	Tendo em mente as possíveis distorções inseridas pelas lentes da câmera na imagem, utiliza-se da equação abaixo para retirar (ou reduzir) o efeito da distorção na imagem. Onde $f_x$ e $f_y$ são os focos da camera e $c_x$ e $c_y$ são os centros ópticos expresso em coordenadas de pixels. A matriz que contém os quatro parâmetros supracitados é chamada de **matriz da câmera**.

$$
\begin{bmatrix}
x \\ 
y \\ 
w   
\end{bmatrix} =  
\begin{bmatrix}
f_x & 0   & c_x \\ 
0   & f_y & c_y \\ 
0   & 0   & 1
\end{bmatrix}
\begin{bmatrix}
X \\ 
Y \\ 
Z   
\end{bmatrix}
$$

​	Portanto, o processo de calibração de câmera consiste em encontrar essas duas matrizes para uma câmera. Na tentativa facilitar esse processo, utiliza-se  de alguns objetos. O mais clássico deles é o tabuleiro de xadrez, o qual apresenta um tamanho prefixado e permite achar pontos conhecidos na imagem e mensurar o quão distorcidos eles estão com relação a referência. Tendo coletado algumas vezes esses pontos de distorção, realiza-se o cálculo das matrizes de distorção e da câmera sobre esses pontos. Quanto mais conjuntos de pontos, mais acurada é a calibração.