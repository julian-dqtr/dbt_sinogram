# Faisceaux harmoniques et conditions de Helgason-Ludwig

*Note de travail pour le mémoire. Objectif : comprendre de A à Z ce qu'est un faisceau
cellulaire, son Laplacien, son énergie de Dirichlet, pourquoi « être harmonique » équivaut
aux conditions de cohérence de Helgason-Ludwig (HLCC), et comment cela justifie une
perspective « Sheaf Attention / Transformer ».*

Toutes les affirmations numériques de ce document sont reproduites par
`scripts/verify_harmonic_sheaves.py` sur la géométrie parallèle du repo (180 vues sur
[-90°, 90°), fenêtre acquise ±25°, graphe kNN k = 12). Les chiffres cités sont ceux de ce
script.

---

## 0. L'idée en dix lignes

1. Un sinogramme parallèle n'est pas une image quelconque : ses moments
   $M_k(\theta)=\int s^k\,p(\theta,s)\,ds$ sont des polynômes trigonométriques très
   contraints (HLCC). $M_0$ est constant, $M_1$ est une sinusoïde pure, $M_2$ contient les
   fréquences 0 et 2, etc.
2. Un **faisceau cellulaire** sur le graphe des vues attache un petit espace vectoriel (une
   *fibre*, ou *stalk*) à chaque vue et une application linéaire (*restriction map*) à chaque
   extrémité d'arête. Il encode une règle : « voici comment comparer les données de deux vues
   voisines ».
3. Une **section globale** est une affectation de données aux vues qui satisfait toutes ces
   règles de comparaison à la fois. L'**énergie de Dirichlet** mesure de combien on les
   viole ; le **Laplacien de faisceau** $L_\mathcal F$ est la forme quadratique associée ;
   son noyau est exactement l'ensemble des sections globales (les signaux *harmoniques*).
4. Si l'on choisit comme restriction maps les rotations $R(-m\theta_v)$, le transport entre
   deux vues est $R(m(\theta_u-\theta_v))$ et les sections globales sont **exactement**
   $a\cos m\theta + b\sin m\theta$ (avec sa quadrature).
5. Donc : *« le moment d'ordre $k$ satisfait les HLCC »* $\iff$ *« il est la somme de
   sections globales des faisceaux de fréquences $m=k, k-2,\dots$ »* $\iff$ *« son énergie de
   Dirichlet de faisceau est nulle »*. C'est un théorème, pas une analogie.
6. **Compléter** les moments manquants revient alors à résoudre un problème de Dirichlet
   discret (*extension harmonique*) : on fixe les vues acquises et on minimise l'énergie sur
   les vues manquantes. Avec les bonnes rotations c'est exact (erreur $5\cdot10^{-15}$) ;
   avec $R=I$, c'est-à-dire le lissage que fait un GCN, l'erreur est de 66 %.
7. Limite honnête : cela ne fixe qu'une quinzaine de nombres par sinogramme (ordres
   $k\le4$), car l'extrapolation depuis ±25° devient instable au-delà. Le faisceau est un
   **régulariseur bas-ordre exact**, pas un moteur de complétion.
8. Un faisceau à connexion plate + attention = **attention rotative (RoPE) sur l'angle
   d'acquisition, à fréquences entières**. C'est la forme naturelle d'un « Sheaf
   Transformer » pour ce problème, et elle supprime la limite de portée angulaire des GNN
   locaux que documente le mémoire.

---

## 1. Rappel : les conditions de Helgason-Ludwig

### 1.1 Cadre

En géométrie parallèle, la vue d'angle $\theta$ mesure

$$p(\theta,s)=\int_{\mathbb R^2} f(\mathbf x)\,\delta\big(s-\mathbf x\cdot \mathbf u_\theta\big)\,d\mathbf x,
\qquad \mathbf u_\theta=(\cos\theta,\ \sin\theta).$$

($\mathbf u_\theta$ est l'axe du détecteur. Le repo utilise $\mathbf u_\theta=(\cos\theta,-\sin\theta)$,
voir `dbt_geometry_2d.py` ; cela change seulement le signe du coefficient de $\sin$ et rien à
ce qui suit.)

### 1.2 Le calcul qui donne tout

Le moment d'ordre $k$ de la vue $\theta$ est

$$M_k(\theta)=\int s^k\,p(\theta,s)\,ds=\int f(\mathbf x)\,(\mathbf x\cdot\mathbf u_\theta)^k\,d\mathbf x
=\sum_{j=0}^{k}\binom kj\,\mu_{k-j,\,j}\ \cos^{k-j}\theta\,\sin^{j}\theta,$$

où $\mu_{a,b}=\int f\,x^a y^b$ sont les moments géométriques de l'objet. Conséquence :
$M_k$ est un **polynôme homogène de degré $k$ en $(\cos\theta,\sin\theta)$**.

- $k=0$ : $M_0(\theta)=\mu_{00}$ = masse totale, **constante**.
- $k=1$ : $M_1(\theta)=\mu_{10}\cos\theta+\mu_{01}\sin\theta$ = masse × projection du centre
  de masse sur le détecteur, **sinusoïde pure**.
- $k=2$ : combinaison de $\cos^2,\ \cos\sin,\ \sin^2$, donc de $1,\ \cos2\theta,\ \sin2\theta$.

### 1.3 Reformulation en harmoniques (celle qui nous sert)

Un produit $\cos^{a}\theta\sin^{b}\theta$ avec $a+b=k$ ne contient que des fréquences
$m\le k$ **de même parité que $k$**. L'espace des polynômes homogènes de degré $k$ est de
dimension $k+1$, et l'espace engendré par $\{\cos m\theta,\sin m\theta : m\equiv k\ (2),\ m\le k\}$
aussi ($1+2\cdot k/2$ si $k$ est pair, $2\cdot(k+1)/2$ sinon). Ces deux espaces coïncident :

$$\boxed{\ \mathcal H_k=\operatorname{span}\{\cos m\theta,\ \sin m\theta\ :\ m=k,k-2,k-4,\dots\ge0\},\qquad \dim\mathcal H_k=k+1\ }$$

**Théorème (Helgason 1965, Ludwig 1966).** Une fonction $p(\theta,s)$ (régulière, à support
compact, paire : $p(\theta+\pi,-s)=p(\theta,s)$) est la transformée de Radon d'un objet
$f$ **si et seulement si** $M_k\in\mathcal H_k$ pour tout $k\ge0$.

Vérification sur le repo (20 fantômes de validation, sans bruit) : la part de $M_k$ hors de
$\mathcal H_k$ vaut 0.06 % ($k=0$), 0.13 % ($k=1$), puis 0.1 à 0.3 % jusqu'à $k=8$. C'est
l'erreur de discrétisation d'ASTRA : la vérité terrain parallèle satisfait les HLCC. (En
fan-beam à détecteur fixe, l'ancien dataset les violait de 35 % et 42 % : ces conditions
n'ont de sens qu'en parallèle, d'où le pivot.)

---

## 2. Faisceaux cellulaires sur un graphe

### 2.1 Définition

Soit $G=(V,E)$ un graphe. Un **faisceau cellulaire** $\mathcal F$ sur $G$ est la donnée de :

- un espace vectoriel $\mathcal F(v)$ pour chaque nœud $v$ (la **fibre**, *stalk*) ;
- un espace vectoriel $\mathcal F(e)$ pour chaque arête $e$ ;
- pour chaque incidence « $v$ est une extrémité de $e$ » (notée $v\trianglelefteq e$), une
  application linéaire $\mathcal F_{v\trianglelefteq e}:\mathcal F(v)\to\mathcal F(e)$, la
  **restriction map**.

Intuition : chaque nœud possède ses données dans son propre système de coordonnées.
L'arête $e=(u,v)$ est un « terrain neutre » où l'on peut comparer $u$ et $v$ :
$\mathcal F_{u\trianglelefteq e}$ et $\mathcal F_{v\trianglelefteq e}$ traduisent chacun
ses données dans le langage commun de l'arête. **Une arête est une contrainte linéaire de
cohérence** : $u$ et $v$ sont d'accord si $\mathcal F_{u\trianglelefteq e}x_u=\mathcal F_{v\trianglelefteq e}x_v$.

Exemple de base : le **faisceau constant** ($\mathcal F(v)=\mathcal F(e)=\mathbb R^d$, toutes
les restriction maps égales à l'identité). « Être d'accord » signifie alors « avoir la même
valeur ». C'est le cadre implicite de tout GNN classique (GCN, GAT...).

### 2.2 Cochaînes, cobord, sections globales

- Une **0-cochaîne** est un choix de donnée en chaque nœud :
  $x=(x_v)_{v\in V}\in C^0=\bigoplus_v\mathcal F(v)$. C'est l'objet qu'un GNN manipule (la
  matrice de features).
- Une **1-cochaîne** vit sur les arêtes : $C^1=\bigoplus_e\mathcal F(e)$.
- Le **cobord** $\delta:C^0\to C^1$ mesure le désaccord sur chaque arête orientée $e=(u\to v)$ :

$$(\delta x)_e=\mathcal F_{v\trianglelefteq e}\,x_v-\mathcal F_{u\trianglelefteq e}\,x_u .$$

- Une **section globale** est une 0-cochaîne sans aucun désaccord :
  $H^0(G;\mathcal F)=\ker\delta$.

Pour le faisceau constant sur un graphe connexe, les sections globales sont les signaux
constants. Pour un faisceau général, ce sont les signaux « constants à traduction près » :
cohérents au sens des restriction maps.

### 2.3 Laplacien de faisceau et énergie de Dirichlet

Le **Laplacien de faisceau** (Hansen et Ghrist 2019) est $L_\mathcal F=\delta^\top\delta$,
matrice symétrique semi-définie positive de taille $\sum_v\dim\mathcal F(v)$. Par blocs :

$$(L_\mathcal F)_{vv}=\sum_{e\ni v}\mathcal F_{v\trianglelefteq e}^\top\mathcal F_{v\trianglelefteq e},
\qquad (L_\mathcal F)_{uv}=-\,\mathcal F_{u\trianglelefteq e}^\top\mathcal F_{v\trianglelefteq e}\quad(e=\{u,v\}).$$

L'**énergie de Dirichlet** d'un signal $x$ est la forme quadratique associée :

$$\boxed{\ E_\mathcal F(x)=x^\top L_\mathcal F\,x=\|\delta x\|^2=\sum_{e=\{u,v\}}w_e\,\big\|\mathcal F_{u\trianglelefteq e}x_u-\mathcal F_{v\trianglelefteq e}x_v\big\|^2\ }$$

(avec des poids d'arêtes $w_e$ éventuels : $L_\mathcal F=\delta^\top W\delta$). Trois faits
découlent immédiatement de $L_\mathcal F=\delta^\top\delta$ :

1. $E_\mathcal F(x)\ge0$, avec égalité **si et seulement si** $\delta x=0$.
2. $\ker L_\mathcal F=\ker\delta=H^0$ : **les signaux d'énergie nulle sont exactement les
   sections globales**. On les appelle signaux *harmoniques* (par analogie avec
   $\Delta f=0$).
3. La **diffusion de faisceau** $\dot x=-L_\mathcal F x$ fait décroître l'énergie et converge
   vers la projection orthogonale de $x(0)$ sur $H^0$ (Bodnar et al. 2022).

Pour le faisceau constant, $L_\mathcal F$ est le Laplacien de graphe usuel,
$E(x)=\sum_{u\sim v}w_{uv}\|x_u-x_v\|^2$ est l'énergie de lissage, et la diffusion est ce que
fait un GCN : elle tire tout vers une constante (*over-smoothing*). Changer de faisceau,
c'est **changer la définition de « lisse »**, donc ce vers quoi la diffusion converge.

### 2.4 Cas des restriction maps orthogonales : connexion et transport parallèle

Si toutes les $\mathcal F_{v\trianglelefteq e}$ sont orthogonales ($O(d)$-fibré), alors

$$\big\|\mathcal F_{u\trianglelefteq e}x_u-\mathcal F_{v\trianglelefteq e}x_v\big\|=\big\|x_u-R_{uv}\,x_v\big\|,\qquad
R_{uv}=\mathcal F_{u\trianglelefteq e}^\top\mathcal F_{v\trianglelefteq e}\in O(d).$$

$R_{uv}$ est le **transport parallèle** de la fibre de $v$ vers celle de $u$. Le Laplacien
devient le **Laplacien de connexion** (Singer et Wu 2012) : $(L)_{vv}=\deg(v)\,I$,
$(L)_{uv}=-w_{uv}R_{uv}$. Sa version normalisée est

$$\Delta_\mathcal F=D^{-1/2}L_\mathcal F D^{-1/2}=I-\underbrace{D^{-1/2}\,(W\circ R)\,D^{-1/2}}_{\hat A_R}.$$

**Lien direct avec le code.** L'agrégation de `ViewGraphConv` calcule
$m_i=\sum_j\hat A_{ij}\,R(\theta_i-\theta_j)\,x_j=(\hat A_R\,x)_i$. Une couche du SNN
applique donc $\hat A_R=I-\Delta_\mathcal F$ : **c'est un pas de diffusion de faisceau**
(suivi d'une conv 1×1 et d'une non-linéarité ; le code ajoute des boucles $w_{ii}=1$ avant de
normaliser, c'est l'astuce de renormalisation $A+I$ de Kipf et Welling). Le GCN applique le même opérateur avec
$R=I$. Le script vérifie que le bloc hors-diagonale de $L_\mathcal F$ vaut bien
$-R(\theta_u-\theta_v)$ et que $x^\top L_\mathcal Fx=\sum_e\|x_u-R_{uv}x_v\|^2$.

### 2.5 Holonomie, platitude, jauge

L'**holonomie** d'un cycle est le produit des transports le long du cycle. Si elle vaut
toujours l'identité, la connexion est **plate**. Un changement de **jauge** est un changement
de repère dans chaque fibre, $y_v=g_v^{-1}x_v$ avec $g_v\in O(d)$ ; il transforme
$R_{uv}$ en $g_u^{-1}R_{uv}g_v$. Une connexion plate sur un graphe connexe peut toujours
être ramenée à $R_{uv}=I$ par un changement de jauge : le faisceau est alors *isomorphe* au
faisceau constant. Ce point est crucial pour interpréter honnêtement le SNN (section 6).

---

## 3. Le faisceau des vues

### 3.1 Construction

Graphe : un nœud par vue, d'angle $\theta_v$. Pour une fréquence entière $m\ge0$, on
définit le faisceau $\mathcal F^{(m)}$ par

$$\mathcal F^{(m)}(v)=\mathcal F^{(m)}(e)=\mathbb R^2,\qquad
\mathcal F^{(m)}_{v\trianglelefteq e}=R(-m\theta_v),\qquad
R(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha\\ \sin\alpha&\cos\alpha\end{pmatrix}.$$

Lecture : la fibre de la vue $v$ est exprimée dans un repère tourné de $m\theta_v$ ;
$R(-m\theta_v)$ la ramène dans le repère fixe du laboratoire, où l'on compare. Le transport
entre deux vues est

$$R_{uv}=R(m\theta_u)\,R(-m\theta_v)=R\big(m(\theta_u-\theta_v)\big).$$

Pour $m=1$ c'est exactement la matrice $R(\theta_i-\theta_j)$ codée en dur dans
`graph_data.build_transport_operators`.

### 3.2 Ses sections globales

$x$ est une section globale ssi $R(-m\theta_u)x_u=R(-m\theta_v)x_v$ sur chaque arête. Sur
un graphe connexe, cette quantité commune est un vecteur constant $c\in\mathbb R^2$, d'où

$$x_v=R(m\theta_v)\,c=\begin{pmatrix}c_1\cos m\theta_v-c_2\sin m\theta_v\\[2pt] c_1\sin m\theta_v+c_2\cos m\theta_v\end{pmatrix}.$$

> **Proposition.** $H^0(G;\mathcal F^{(m)})$ est de dimension 2. La première composante
> d'une section globale parcourt exactement $\{a\cos m\theta+b\sin m\theta\}$ ; la seconde
> est sa **quadrature** (la même sinusoïde déphasée de $\pi/2m$).

Vérification numérique (graphe kNN $k=12$ du repo, 180 vues, $m=0..3$) :
$\dim\ker L_{\mathcal F^{(m)}}=2$ et les sections prédites sont dans le noyau à $10^{-13}$
près. En notation complexe c'est encore plus court : fibre $\mathbb C$, restriction
$e^{-im\theta_v}$, sections $z_v=c\,e^{im\theta_v}$, et le signal réel est $\mathrm{Re}(z_v)$.

Pourquoi une fibre de dimension 2 pour un moment scalaire ? Parce qu'une sinusoïde de phase
inconnue a deux degrés de liberté $(a,b)$ : il faut transporter le couple (signal,
quadrature) pour que le transport soit une simple rotation. C'est la même raison pour
laquelle on représente un signal oscillant par un phaseur complexe.

### 3.3 Refermer le cercle : la règle de parité est une condition d'holonomie

Les vues couvrent $[-90°,90°)$, mais physiquement $\theta$ et $\theta+\pi$ regardent les
mêmes droites avec le détecteur retourné : $p(\theta+\pi,s)=p(\theta,-s)$, donc
$M_k(\theta+\pi)=(-1)^kM_k(\theta)$. Si l'on referme le chemin des vues en cercle (arête
entre 89° et $-90°+180°$), faire un tour complet multiplie la fibre de $\mathcal F^{(m)}$
par $R(m\pi)=(-1)^m$ et le moment d'ordre $k$ par $(-1)^k$. Une section globale non nulle
n'existe que si l'holonomie totale est triviale :

$$(-1)^{k+m}=+1\iff m\equiv k\pmod 2 .$$

Vérifié numériquement : pour $k$ pair, $\dim H^0=2,0,2,0$ pour $m=0,1,2,3$ ; pour $k$
impair, $0,2,0,2$. **La règle de parité des HLCC est donc la condition d'existence de
sections globales sur le cercle des vues.** C'est aussi le seul endroit du problème où
apparaît une holonomie non triviale (le retournement du détecteur : le fibré des lignes
détecteur au-dessus de $\mathbb{RP}^1$ est un ruban de Möbius), et donc le seul endroit où
un faisceau n'est pas réductible à un simple encodage positionnel.

---

## 4. Le théorème : HLCC $\iff$ harmonicité

### 4.1 Énoncé

Pour un ordre maximal $K$, on forme le faisceau somme directe qui contient une copie de
$\mathcal F^{(m)}$ par couple (ordre $k$, harmonique $m$ autorisée) :

$$\mathcal F_K=\bigoplus_{k=0}^{K}\ \bigoplus_{\substack{0\le m\le k\\ m\equiv k\ (2)}}\mathcal F^{(m)} .$$

Un champ de moments $\{M_k(\theta_v)\}_{k\le K,\ v\in V}$ admet un **relèvement** $x$ dans
$\mathcal F_K$ si $M_k(\theta_v)=\sum_m\big[x_v^{(k,m)}\big]_1$ (somme des premières
composantes).

> **Théorème.** Les moments $M_0,\dots,M_K$ d'un sinogramme échantillonné satisfont les HLCC
> jusqu'à l'ordre $K$ **si et seulement si** ils admettent un relèvement qui est une section
> globale de $\mathcal F_K$, c'est-à-dire ssi
> $$\min_{x\ \text{relève}\ M}\ E_{\mathcal F_K}(x)=0 .$$

*Preuve.* ($\Leftarrow$) Si $x$ est une section globale, chaque $x^{(k,m)}_v=R(m\theta_v)c^{(k,m)}$
a pour première composante $a\cos m\theta_v+b\sin m\theta_v$ ; leur somme sur $m\equiv k$,
$m\le k$ est dans $\mathcal H_k$. ($\Rightarrow$) Si $M_k\in\mathcal H_k$, on l'écrit
$\sum_m(a_m\cos m\theta+b_m\sin m\theta)$ et on relève chaque terme avec sa quadrature
($c^{(k,m)}=(a_m,-b_m)$) : on obtient une section globale, d'énergie nulle. $\blacksquare$

Remarque : l'énergie partielle $\tilde E_k(M_k)=\min_{x\text{ relève }M_k}E(x)$ est une forme
quadratique en $M_k$ (un complément de Schur de $L_\mathcal F$) dont le noyau est exactement
$\mathcal H_k$.

### 4.2 La perte HLCC du repo est déjà une énergie de faisceau

`HelgasonLudwigLoss` calcule :

- **Ordre 0** : $\mathrm{Var}_v\big(M_0(\theta_v)\big)$. Or l'identité
  $\sum_{u<v}(x_u-x_v)^2=V\sum_v(x_v-\bar x)^2$ donne
  $\mathrm{Var}(M_0)=\frac{1}{V(V-1)}\sum_{u<v}\big(M_0(\theta_u)-M_0(\theta_v)\big)^2$ :
  c'est l'énergie de Dirichlet du **faisceau constant** $\mathcal F^{(0)}$ sur le graphe
  complet des vues.
- **Ordre 1** : $\|(I-P)M_1\|^2/V$ où $P$ projette sur $\operatorname{span}\{\cos\theta,\sin\theta\}$ :
  c'est la distance au carré de $M_1$ à $\ker L_{\mathcal F^{(1)}}$ (premières composantes).

Les deux pénalités et les énergies de faisceau $\tilde E_0,\tilde E_1$ ont **le même ensemble
de zéros** ; elles diffèrent seulement par la pondération des violations (le projecteur
pénalise tous les modes non harmoniques de la même façon, l'énergie de Dirichlet sur un
graphe local pénalise davantage les violations de haute fréquence angulaire). Le formalisme
des faisceaux donne donc (i) une lecture géométrique de la perte existante et (ii) sa
généralisation immédiate à tout ordre $K$ : $\sum_{k\le K}\lambda_k\tilde E_k$.

---

## 5. Compléter = extension harmonique (problème de Dirichlet discret)

### 5.1 Principe

Partitionnons les vues en acquises $A$ et manquantes $U$. On fixe $x_A$ et on cherche
$x_U$ qui minimise l'énergie :

$$\min_{x_U}\ \begin{pmatrix}x_A\\x_U\end{pmatrix}^{\!\top}\begin{pmatrix}L_{AA}&L_{AU}\\L_{UA}&L_{UU}\end{pmatrix}\begin{pmatrix}x_A\\x_U\end{pmatrix}
\quad\Longrightarrow\quad \boxed{\ L_{UU}\,x_U=-L_{UA}\,x_A\ }$$

C'est l'analogue discret de « résoudre $\Delta f=0$ avec condition au bord ». $L_{UU}$ est
inversible dès que chaque composante connexe de $U$ touche $A$. Si $x_A$ est la restriction
d'une section globale, l'unique minimiseur **est** cette section (énergie nulle).

### 5.2 L'expérience qui résume tout

Sur un fantôme du jeu de validation, on relève $M_1$ en section de $\mathcal F^{(1)}$, on
ne garde que les 51 vues acquises et on résout le système ci-dessus sur le graphe kNN
$k=12$ :

| Faisceau utilisé pour l'extension | Erreur relative sur le wedge |
|---|---|
| $\mathcal F^{(1)}$, transport $R(\theta_u-\theta_v)$ (SNN) | $4.6\cdot10^{-15}$ |
| faisceau constant, $R=I$ (lissage GCN) | $0.66$ |

Avec le bon faisceau, la sinusoïde est prolongée exactement sur les 129 vues manquantes,
**avec un opérateur purement local**. Avec le faisceau constant, l'extension harmonique est
l'interpolation la plus lisse possible, qui aplatit la sinusoïde. C'est la différence
conceptuelle entre les deux modèles de l'ablation : le GCN diffuse vers « constant », le SNN
diffuse vers « harmonique de fréquence $m$ ».

### 5.3 Si la quadrature n'est pas mesurée

En pratique on mesure $M_k$ (scalaire) et pas sa quadrature. L'extension harmonique devient
un moindres carrés sur les coefficients $(a_m,b_m)$ ajustés sur $A$ puis évalués sur $U$ :
pour $k=1$ c'est la « régression HL » de Huang et al. (2017).

### 5.4 Limite fondamentale : le conditionnement

Prolonger un polynôme trigonométrique connu sur 50° à un arc de 180° est une continuation
analytique, mal conditionnée. Mesuré sur 20 fantômes (ajustement sur ±25°, erreur sur le
wedge) :

| Ordre $k$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Erreur médiane d'extension | 0.05 % | 0.11 % | 0.26 % | 1.1 % | 5.4 % | 13 % | 90 % | 406 % |
| Conditionnement | 1 | 3.8 | 16 | 69 | 290 | 1.3e3 | 5.4e3 | 2.3e4 |

Le conditionnement est multiplié par environ 4.4 à chaque ordre : avec une erreur de
discrétisation de $10^{-3}$, l'extension est fiable jusqu'à $k\approx4$. Cela fixe
$\sum_{k\le4}(k+1)=15$ nombres réels par sinogramme, face à $129\times128=16\,512$ pixels
inconnus. C'est la version « moments » du caractère mal posé de la tomographie à angle
limité (Davison 1983, Louis 1986, Natterer 1986). **Le faisceau est un régulariseur
bas-ordre exact, jamais le moteur de la complétion** : les hautes fréquences doivent venir
d'un a priori appris.

---

## 6. Ce que fait (et ne fait pas) le SNN du repo

À écrire tel quel dans le mémoire, pour désamorcer les objections :

1. **Ce qu'il est.** Un réseau sur le graphe des vues dont chaque couche applique
   $\hat A_R=I-\Delta_\mathcal F$, un pas de diffusion du Laplacien de connexion normalisé de
   $\mathcal F^{(1)}$, à chaque paire de canaux. C'est un cas particulier des sheaf NN à
   Laplacien de connexion (Barbero et al. 2022) et des Bundle NN (Bamberger et al.), avec une
   connexion **fixée par la géométrie d'acquisition** au lieu d'être apprise.
2. **La connexion est plate.** $R(\theta_u-\theta_v)=R(\theta_u)R(\theta_v)^\top$ : holonomie
   triviale. Par le changement de jauge $y_v=R(-\theta_v)x_v$, la partie linéaire du SNN
   devient celle du GCN. La différence entre les deux modèles est donc **le repère dans
   lequel agissent les non-linéarités et les convolutions**, c'est-à-dire une façon
   structurée d'injecter l'angle absolu (un encodage positionnel rotatif). C'est une
   hypothèse testable, et c'est exactement ce que teste l'ablation GCN vs SNN.
3. **Les fibres ne sont pas des moments.** La rotation est appliquée à des canaux bruts,
   identiquement aux 128 pixels du détecteur. Or le théorème de la section 4 porte sur des
   *moments* (intégrales le long du détecteur), avec *une fréquence par ordre*. Le SNN
   actuel ne transporte exactement que le contenu de fréquence angulaire $m=1$ ; le contenu
   dominant ($m=0$) serait mieux transporté par l'identité.
4. **La portée est locale.** Une couche propage l'information de $k/2=6$ vues. La vue
   manquante la plus éloignée est à 65° de la fenêtre acquise : il faut au moins 11 couches
   pour l'atteindre (6 couches : 36°, 12 : 72°, 18 : 108°). Le test
   `test_angular_receptive_field_is_exactly_layers_times_half_k` montre que cette borne est
   exacte, quels que soient les poids. Et la diffusion est lente : la première valeur propre
   non nulle du Laplacien du graphe kNN (degré 12) vaut $0.028$, soit un trou spectral
   normalisé de l'ordre de $2\cdot10^{-3}$.

Conclusion défendable : *le SNN implémente un pas de diffusion de faisceau correct, mais sur
des fibres qui ne sont pas celles pour lesquelles le faisceau est exact, et avec une portée
limitée par la localité du graphe.* Les deux limites pointent vers la même perspective.

---

## 7. Perspective : « Sheaf Attention / Transformer »

### 7.1 De la diffusion de faisceau à l'attention de faisceau

Une couche de sheaf attention (Barbero et al. 2022, *Sheaf Attention Networks*) remplace les
poids fixes $\hat A_{ij}$ par des poids dépendant des données, en gardant le transport :

$$x_i'=\sum_{j}\alpha_{ij}(x)\;R_{ij}\,V x_j,\qquad
\alpha_{ij}=\operatorname{softmax}_j\!\Big(\tfrac{1}{\sqrt d}\,q_i^\top R_{ij}\,k_j\Big).$$

Sur le graphe **complet** des vues, la portée angulaire devient infinie en une seule couche :
la limite de localité étudiée dans le mémoire disparaît par construction ($V=180$ jetons,
l'attention en $O(V^2)$ est négligeable).

### 7.2 Connexion plate + attention = attention rotative (RoPE) sur l'angle

Comme la connexion est plate, $R_{ij}=R(m\theta_i)R(m\theta_j)^\top$, donc

$$q_i^\top R_{ij}\,k_j=\big(R(m\theta_i)^\top q_i\big)^{\!\top}\big(R(m\theta_j)^\top k_j\big).$$

Il suffit de tourner requêtes et clés par leur angle **absolu** pour que le produit scalaire
ne dépende que de l'angle **relatif**. C'est exactement le mécanisme des *rotary position
embeddings* (Su et al. 2021), où la « position » est ici l'angle d'acquisition $\theta$ et
où chaque paire de canaux $f$ porte une fréquence $m_f$. Deux différences avec le RoPE
standard, toutes deux dictées par la physique :

- **fréquences entières** $m_f\in\{0,1,\dots,K\}$ (et non géométriques) : ce sont les seules
  pour lesquelles les sections globales sont les harmoniques des HLCC, et les seules
  compatibles avec la périodicité en $\theta$ ;
- **fermeture de Möbius** : l'arête entre $+90°$ et $-90°$ doit retourner le détecteur
  ($s\to-s$), ce qui est la seule holonomie non triviale du problème (section 3.3).

On obtient ainsi une lecture précise du mot « sheaf » : *le transport parallèle du faisceau
$\bigoplus_f\mathcal F^{(m_f)}$ est le RoPE angulaire à fréquences entières*. Propriété
utile : les logits ne dépendent que de $\theta_i-\theta_j$, donc l'opérateur est équivariant
aux rotations de l'objet (qui décalent le sinogramme en $\theta$).

### 7.3 Architecture proposée

1. **Jetons** = vues. Plongement d'une vue : features apprises de la ligne détecteur (CNN 1D)
   **plus** ses moments bas-ordre $M_0..M_K$ en base orthogonale (Legendre ou Tchebychev, mieux
   conditionnés que les monômes $s^k$).
2. **Blocs d'attention de faisceau** : canaux « moments » transportés à la fréquence
   prescrite par la section 4 ($m\equiv k$, $m\le k$) ; canaux « contenu » à $m=0$
   (attention ordinaire), responsables des hautes fréquences.
3. **Tête harmonique** : soit une projection dure des moments prédits sur $\mathcal H_k$
   (projecteur linéaire fermé, différentiable), soit la perte
   $\sum_{k\le K}\lambda_k\tilde E_k$ de la section 4.2, qui généralise la perte HLCC
   actuelle. D'après 5.4, $K=4$ est le maximum raisonnable pour une fenêtre de ±25°.
4. **Consistance aux données** inchangée (masque doux à l'intérieur de la fenêtre).

### 7.4 Ablations qui rendraient la contribution vérifiable

$R=I$ (Transformer sans position) ; encodage sinusoïdal additif de $\theta$ ; RoPE à
fréquences entières (faisceau plat) ; idem avec fermeture de Möbius ; connexion apprise mais
contrainte à $SO(2)$ ($z=e^{i(m\Delta\theta+\delta)}$) ; avec et sans tête harmonique. Métriques :
erreur par bande angulaire, erreur sur les moments du wedge, énergie de Dirichlet
$\tilde E_k$ de la sortie.

### 7.5 Ce qu'il ne faut pas promettre

- Aucune architecture ne recrée l'information absente des données : le faisceau contraint
  ~15 nombres, le reste est un a priori appris sur la distribution des objets.
- Une connexion plate n'apporte aucun des bénéfices « topologiques » des sheaf diffusions
  (hétérophilie, anti-oversmoothing via holonomie) : l'argument ici est **géométrique**
  (bon repère, bonnes harmoniques), pas topologique. Le seul ingrédient topologique réel est
  la torsion de Möbius.
- En fan-beam à détecteur fixe, rien de tout cela ne s'applique sans rebinning : il n'existe
  pas d'action de $SO(2)$ reliant les vues, et les HLCC parallèles sont fausses.

---

## 8. Formulations prêtes à l'emploi pour le mémoire

**Positionnement.** « Nous modélisons le sinogramme comme un signal sur le graphe des vues,
muni d'un faisceau cellulaire à fibres $\mathbb R^2$ dont les restriction maps sont les
rotations $R(-m\theta_v)$ fixées par les angles d'acquisition. Ce faisceau définit une
connexion $SO(2)$ plate ; son Laplacien est un Laplacien de connexion au sens de Singer et
Wu. »

**Résultat théorique.** « Les sections globales de ce faisceau sont exactement les
harmoniques angulaires de fréquence $m$. Par conséquent, un sinogramme parallèle satisfait
les conditions de Helgason-Ludwig jusqu'à l'ordre $K$ si et seulement si son champ de moments
se relève en une section globale du faisceau $\mathcal F_K=\bigoplus_{k\le K}\bigoplus_{m\equiv k,\,m\le k}\mathcal F^{(m)}$,
c'est-à-dire si son énergie de Dirichlet de faisceau est nulle. La règle de parité
$m\equiv k\pmod 2$ s'interprète comme une condition d'holonomie sur le cercle des vues
refermé par l'identification $p(\theta+\pi,s)=p(\theta,-s)$. »

**Limite documentée.** « La complétion des moments est une extension harmonique, exacte mais
mal conditionnée : depuis une fenêtre de ±25°, seuls les ordres $k\le4$ (15 degrés de liberté)
sont extrapolables de façon stable. Le faisceau agit donc comme un régulariseur bas-ordre. Par
ailleurs, un réseau à passage de messages local de profondeur $L$ sur un graphe kNN a une
portée angulaire exactement égale à $L\cdot k/2$ vues, ce qui borne a priori la région du
wedge qu'il peut informer. »

**Perspective.** « Une attention de faisceau sur le graphe complet des vues lève la limite de
portée. La connexion étant plate, elle se réduit à une attention rotative dont la position
est l'angle d'acquisition et dont les fréquences sont entières ; ces fréquences sont
précisément celles des conditions de Helgason-Ludwig, ce qui permet de transporter exactement
les moments bas-ordre et d'imposer leur harmonicité par une projection ou une énergie de
Dirichlet. »

---

## 9. Glossaire

| Terme | Signification |
|---|---|
| Fibre (*stalk*) $\mathcal F(v)$ | espace vectoriel des données du nœud $v$ |
| Restriction map $\mathcal F_{v\trianglelefteq e}$ | traduction des données de $v$ dans l'espace de l'arête $e$ |
| 0-cochaîne | un signal : une donnée par nœud (la matrice de features) |
| Cobord $\delta$ | désaccord sur chaque arête |
| Section globale, $H^0=\ker\delta$ | signal sans aucun désaccord |
| Laplacien de faisceau $L_\mathcal F=\delta^\top\delta$ | généralise le Laplacien de graphe |
| Énergie de Dirichlet $x^\top L_\mathcal Fx$ | somme des désaccords au carré |
| Harmonique | d'énergie nulle, dans $\ker L_\mathcal F$ |
| Extension harmonique | minimiser l'énergie à données fixées sur une partie des nœuds |
| Transport parallèle $R_{uv}$ | $\mathcal F_{u\trianglelefteq e}^\top\mathcal F_{v\trianglelefteq e}$ (cas orthogonal) |
| Holonomie | produit des transports le long d'un cycle |
| Connexion plate | holonomie triviale ; équivalente par jauge au faisceau constant |
| Jauge | choix d'un repère dans chaque fibre |

## 10. Références

*(Détails bibliographiques à vérifier avant citation.)*

- S. Helgason, *The Radon transform on Euclidean spaces, compact two-point homogeneous spaces and Grassmann manifolds*, Acta Math., 1965.
- D. Ludwig, *The Radon transform on Euclidean space*, Comm. Pure Appl. Math., 1966.
- F. Natterer, *The Mathematics of Computerized Tomography*, Wiley, 1986.
- M. E. Davison, *The ill-conditioned nature of the limited angle tomography problem*, SIAM J. Appl. Math., 1983.
- A. K. Louis, *Incomplete data problems in X-ray computerized tomography I*, Numer. Math., 1986.
- Y. Huang et al., *Restoration of missing data in limited angle tomography based on Helgason-Ludwig consistency conditions*, Biomed. Phys. Eng. Express, 2017.
- J. Curry, *Sheaves, cosheaves and applications*, thèse, Univ. of Pennsylvania, 2014.
- J. Hansen, R. Ghrist, *Toward a spectral theory of cellular sheaves*, J. Appl. Comput. Topology, 2019.
- J. Hansen, T. Gebhart, *Sheaf neural networks*, NeurIPS TDA workshop, 2020.
- C. Bodnar et al., *Neural sheaf diffusion: a topological perspective on heterophily and oversmoothing in GNNs*, NeurIPS, 2022.
- F. Barbero et al., *Sheaf neural networks with connection Laplacians*, ICML TAG workshop, 2022.
- F. Barbero et al., *Sheaf attention networks*, NeurIPS NeurReps workshop, 2022.
- J. Bamberger et al., *Bundle neural networks for message diffusion on graphs*, 2024.
- A. Singer, H.-T. Wu, *Vector diffusion maps and the connection Laplacian*, Comm. Pure Appl. Math., 2012.
- J. Su et al., *RoFormer: enhanced transformer with rotary position embedding*, 2021.
- T. Kipf, M. Welling, *Semi-supervised classification with graph convolutional networks*, ICLR, 2017.
