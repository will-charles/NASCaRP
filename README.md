<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<h3 align="center">NASCaRP</h3>

  <p align="center">
    A neural accelerated sampling method for fast calculation of radiative processes.
    <a href="https://arxiv.org/abs/2406.19385"><strong>Read the paper »</strong></a>
    <br />
    <br />
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

This is an implementation of the numerical method described in **A Machine Learning Method for Monte Carlo Calculations of Radiative Processes** by William Charles and Alexander Y. Chen.

Included here is a minimal working example for training a neural network from scratch to approximate the inverse Compton radiative interaction, the parameters of our pre-trained network, 
and an implementation of neural network evaluation in CUDA using low-level WMMA operations directly.

<!-- GETTING STARTED -->
### Prerequisites

Training and testing the network in Python requires PyTorch and NumPy. Visualizations require Matplotlib. GPU implementation requires CUDA with cuBLAS.

## Getting Started

1. Clone the repo
   ```sh
   git clone git@github.com:will-charles/NASCaRP.git
   ```
2. Training and testing can be run directly in the Jupyter notebook provided.
3. Compile the GPU evaluation with:
   ```sh
   nvcc -x cu -lcublasLt -arch=<COMPUTE ARCHITECTURE> -o sk_5 sk_5.cpp
   ```
   You will need to specify the GPU architecture. This implementation makes use of Tensor Cores on NVIDIA GPUs.
4. Run the GPU evaluation after compilation with:
   ```sh
   ./sk_5 <log (electron Lorentz factor)> <log (dimensionless photon energy)> <output file name (.txt)>
   ```
   The default setting is for the script to merely time the sampling process. If you want to record the output of sampled values, please change this line in the script from 0 to 1:
   ```sh
   int write_output = 1;
   ```
   To change the number of samples computed, please change this line in the script to the desired number of samples:
   ```sh
   constexpr size_t M = 1ul<<24;
   ```

<!-- CONTRIBUTING -->
## Contributing

Any contributions are greatly appreciated. If you have a suggestion that would make this better, 
please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
## Contact

William Charles - charles.william@wustl.edu

Project Link: [https://github.com/will-charles/NASCaRP](https://github.com/will-charles/NASCaRP)


<p align="right">(<a href="#readme-top">back to top</a>)</p>
