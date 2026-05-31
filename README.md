# TinyEKF 🚀
**A lightweight, zero-allocation, and symbolic-based Extended Kalman Filter for microcontrollers.**

O TinyEKF utiliza álgebra simbólica via Python (SymPy) para calcular Jacobianos e simplificar equações não lineares *antes* da compilação. Ele entrega um código C++ otimizado, sem loops e sem alocação dinâmica, pronto para rodar em modelos físicos complexos.

## ✨ Por que TinyEKF?
* **Zero Álgebra no MCU:** Derivadas (Jacobianos) e inversões de matrizes são resolvidas algebricamente no Python. O microcontrolador executa apenas aritmética de ponto flutuante pura.
* **Foco em Não Linearidade:** Suporte nativo para modelos físicos não lineares ($f(x, u)$ e $h(x)$). O Python calcula automaticamente as derivadas parciais.
* **Zero Loops (Loop Unrolling):** Código "achatado" para máxima performance. Sem `for` ou `while` que drenem o seu clock.
* **Zero Alocação Dinâmica:** Memória 100% estática. Nada de `new` ou `malloc()`. Segurança total para sistemas críticos.
* **Framework Agnostic:** Gera código C++ padrão compatível com Arduino, ESP-IDF, STM32 e outros.

## 🛠 Como funciona?

1. **O Gerador (Python):** Você define as equações não lineares do seu sistema ($f$ e $h$) no seu notebook ou script. O script utiliza o `sympy.jacobian()` para derivar as matrizes e `sympy.cse()` para otimizar subexpressões, eliminando cálculos redundantes.
2. **A Otimização:** O script gera o `TinyLKF.cpp` com todas as constantes pré-calculadas e simplificadas.
3. **O Alvo (C++):** Você integra os arquivos `TinyLKF.h` e `TinyLKF.cpp` ao seu projeto. O seu MCU fará apenas o trabalho pesado da predição e atualização, sem precisar de bibliotecas pesadas de álgebra linear.

## 🚀 Primeiros Passos
1. Edite o arquivo `optimizer.py` definindo o seu modelo não linear (`f` e `h`).
2. Execute o script: `python optimizer.py`.
3. O `TinyLKF.cpp` será atualizado automaticamente com o seu novo modelo.
4. Compile o seu projeto C++.