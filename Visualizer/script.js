// DOM references
const x1Input = document.getElementById("x1");
const x2Input = document.getElementById("x2");
const targetInput = document.getElementById("target");
const lrInput = document.getElementById("lr");
const activationSelect = document.getElementById("activation");

const lossDiv = document.getElementById("loss");
const gradientsDiv = document.getElementById("gradients");

const h1Node = document.getElementById("h1");
const h2Node = document.getElementById("h2");
const outNode = document.getElementById("outNode");

const backward1 = document.getElementById("b1");
const backward2 = document.getElementById("b2");

const graph = document.getElementById("graph");
const ctx = graph.getContext("2d");

const weightLabels = [
  document.getElementById("w1"),
  document.getElementById("w2"),
  document.getElementById("w3"),
  document.getElementById("w4"),
  document.getElementById("w5"),
  document.getElementById("w6")
];

// Network
let network;
let losses=[];
let auto=false;
let cache={};

function initNetwork(){
  network={
    W1:[[0.5,-0.3],[0.8,0.2]],
    W2:[0.4,-0.7]
  };
  losses=[];
  drawGraph();
  updateWeights();
  lossDiv.innerText="-";
  gradientsDiv.innerText="-";
}
initNetwork();

function activate(x){
  if(activationSelect.value==="relu") return Math.max(0,x);
  return 1/(1+Math.exp(-x));
}

function derivative(x){
  if(activationSelect.value==="relu") return x>0?1:0;
  const s=1/(1+Math.exp(-x));
  return s*(1-s);
}

function forwardPass(){
  const x1=+x1Input.value;
  const x2=+x2Input.value;
  const target=+targetInput.value;

  const z1=network.W1[0][0]*x1+network.W1[0][1]*x2;
  const z2=network.W1[1][0]*x1+network.W1[1][1]*x2;

  const a1=activate(z1);
  const a2=activate(z2);

  const z3=network.W2[0]*a1+network.W2[1]*a2;
  const out=activate(z3);

  const loss=0.5*(target-out)**2;

  cache={x1,x2,z1,z2,a1,a2,z3,out,target,loss};
  lossDiv.innerText=loss.toFixed(6);

  h1Node.setAttribute("fill",`rgba(59,130,246,${a1})`);
  h2Node.setAttribute("fill",`rgba(59,130,246,${a2})`);
  outNode.setAttribute("fill",`rgba(34,197,94,${out})`);

  return cache;
}

function trainStep(){
  forwardPass();

  const lr=+lrInput.value;
  const error=cache.out-cache.target;

  const dZ3=error*derivative(cache.z3);
  const dW2_0=dZ3*cache.a1;
  const dW2_1=dZ3*cache.a2;

  const dZ1=dZ3*network.W2[0]*derivative(cache.z1);
  const dZ2=dZ3*network.W2[1]*derivative(cache.z2);

  network.W2[0]-=lr*dW2_0;
  network.W2[1]-=lr*dW2_1;
  network.W1[0][0]-=lr*dZ1*cache.x1;
  network.W1[0][1]-=lr*dZ1*cache.x2;
  network.W1[1][0]-=lr*dZ2*cache.x1;
  network.W1[1][1]-=lr*dZ2*cache.x2;

  gradientsDiv.innerHTML=
    `dW2: [${dW2_0.toFixed(4)}, ${dW2_1.toFixed(4)}]`;

  backward1.style.visibility="visible";
  backward2.style.visibility="visible";

  losses.push(cache.loss);
  drawGraph();
  updateWeights();
}

function drawGraph(){
  ctx.clearRect(0,0,240,150);
  ctx.beginPath();
  losses.forEach((l,i)=>{
    const x=i*4;
    const y=150-l*200;
    if(i===0) ctx.moveTo(x,y);
    else ctx.lineTo(x,y);
  });
  ctx.strokeStyle="#22c55e";
  ctx.stroke();
}

function updateWeights(){
  weightLabels[0].textContent=network.W1[0][0].toFixed(2);
  weightLabels[1].textContent=network.W1[0][1].toFixed(2);
  weightLabels[2].textContent=network.W1[1][0].toFixed(2);
  weightLabels[3].textContent=network.W1[1][1].toFixed(2);
  weightLabels[4].textContent=network.W2[0].toFixed(2);
  weightLabels[5].textContent=network.W2[1].toFixed(2);
}

function autoTrain(){
  if(auto){auto=false;return;}
  auto=true;
  function loop(){
    if(!auto) return;
    trainStep();
    setTimeout(loop,200);
  }
  loop();
}

// Button bindings
document.getElementById("forwardBtn").onclick=forwardPass;
document.getElementById("trainBtn").onclick=trainStep;
document.getElementById("autoBtn").onclick=autoTrain;
document.getElementById("resetBtn").onclick=initNetwork;
