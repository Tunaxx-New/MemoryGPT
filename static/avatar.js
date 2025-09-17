import {TEXT2MOTION_API_KEY} from './config.js';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin} from '@pixiv/three-vrm';

// Scene and Camera
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 1, 2); // move camera back and up to see model
camera.lookAt(0, 1, 0); // look at typical VRM model center

// Renderer
let renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(window.innerWidth, window.innerHeight);

const container = document.getElementById('avatar-container');
container.appendChild(renderer.domElement);

// Lights
const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(0, 10, 10);
scene.add(directionalLight);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // soft light
scene.add(ambientLight);

// Load VRM
let vrm = null;
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

let idleAnimationData = null;
let animationData = null;
let nameMapData = {};

function mapBoneName(name) {
    // Remove digits
    name = name.replace(/\d+/g, '');

    // Handle Left/Right suffix
    let prefix = '';
    if (name.endsWith('L')) {
        prefix = 'Left';
        name = name.slice(0, -1);
    } else if (name.endsWith('R')) {
        prefix = 'Right';
        name = name.slice(0, -1);
    }

    // Split by underscore and capitalize each part
    const parts = name.split('_').map(part => part.charAt(0).toUpperCase() + part.slice(1));

    // Join parts
    let mappedName = parts.join('');

    // Add Left/Right prefix
    mappedName = prefix + mappedName;

    // Replace "Upper" with "Up"
    mappedName = mappedName.replace(/Upper/g, 'Up');
    mappedName = mappedName.replace(/Lower/g, '');

    // Add mixamorig prefix
    mappedName = 'mixamorig' + mappedName;
    console.log(mappedName)
    return mappedName;
}


function buildBoneTree(bone, maxDepth = 12, currentDepth = 0) {
    if (!bone || currentDepth > maxDepth) return null;

    // Ensure world matrices are up-to-date
    bone.updateMatrixWorld(true);
    const cleanName = bone.name;
    const changedName = mapBoneName(bone.name);

    nameMapData[changedName] = bone.name;

    const node = {
        name: changedName || "unnamed",
        matrix: Array.from(bone.matrixWorld.elements),
        children: []
    };
    // Recurse for children
    bone.children.forEach(child => {
        const childNode = buildBoneTree(child, maxDepth, currentDepth + 1);
        if (childNode) node.children.push(childNode);
    });

    return node;
}

function findHumanBone(vrm, name) {
    for (const boneKey in vrm.humanoid._rawHumanBones.humanBones) {
        const bone = vrm.humanoid._rawHumanBones.humanBones[boneKey].node;
        if (!bone) continue;
        if (bone.name === name) return bone;
    }
    return null;
}

function applyAnimation(vrm, animationData, nameMap, t = 0, duration = null) {
    if (!animationData || !animationData.bones) return;
    const bones = animationData.bones; // { boneName: [[matrix16 at frame0], [matrix16 at frame1], ...] }
    for (const cleanName in bones) {
        let originalName = nameMapData[cleanName]; // restore original
        const bone = findHumanBone(vrm, originalName);

        if (!bone) continue;

        // lower_armL
        // Apply matrix for current frame (for simplicity, first frame)
        const boneData = bones[cleanName]; // assume bones[cleanName] is an array of matrices
        if (!boneData.position && !boneData.rotation) continue;

        let time_keys = boneData.position
        if (!time_keys) time_keys = boneData.rotation
        const timestamps = Object.keys(time_keys).map(parseFloat);

        // Find the closest timestamp <= t
        let currentTime = t;
        if (duration) currentTime = Math.min(t, duration); // clamp to duration// t normalized 0-1 -> seconds
        let closestTime = timestamps[0];
        for (let ts of timestamps) {
            if (ts <= currentTime) closestTime = ts;
            else break;
        }
        const key = closestTime.toString();

        if (boneData.position && bone.position) {
            const pos = boneData.position[key];
            if (pos) bone.position.fromArray(pos);
        }
        if (boneData.rotation && bone.quaternion) {
            const rot = boneData.rotation[key];
            if (rot) bone.quaternion.fromArray(rot);
        }

        bone.updateMatrix();
        bone.matrixWorldNeedsUpdate = true;
    }
    vrm.scene.updateMatrixWorld(true);
}

async function request_animation(prompt) {
    if (!vrm || !vrm.scene || !vrm.humanoid) {
        console.error("VRM not loaded properly");
        return;
    }

    const rootBone = vrm.humanoid.getRawBoneNode('hips');
    if (!rootBone) {
        console.error("Root bone not found");
        return;
    }

    const skeletonTree = buildBoneTree(rootBone);

    // --- Get all standard humanoid bones and data ---
    const humanoid = vrm.humanoid;
    const bones = {};
    for (const boneName in humanoid.humanBones) {
        const humanBone = humanoid.humanBones[boneName];
        if (humanBone) {
            bones[boneName] = {}
            bones[boneName]['name'] = boneName;
            bones[boneName]['matrix'] = humanBone.node.matrixWorld.elements;
        }
    }
    const modelMatrix = vrm.scene.matrixWorld.elements;


    // Request to API
    const body = {
        prompt: prompt,
        target_skeleton: {
            world_matrix: modelMatrix,
            root: skeletonTree
        }
    };

    try {
        const response = await fetch('https://api.text2motion.ai/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-apikey': TEXT2MOTION_API_KEY
            },
            body: JSON.stringify(body)
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        const data = await response.json();
        return JSON.parse(data.result);
    } catch (err) {
        console.error(err);
    }
}

async function requestAndPlayAnimation(prompt, isIdle = false) {
    if (!vrm) return;

    // Request animation from API
    const animation = await request_animation(prompt);
    animationData = animation
    if (!animationData) return;

    if (isIdle) {
        idleAnimationData = animationData
    }

    // Start animation loop

    clock.start(); // reset clock
    animateVRM();
}

function getAnimationDuration(animationData) {
    let maxTime = 0;

    for (const boneName in animationData.bones) {
        const boneData = animationData.bones[boneName];
        const times = [];

        if (boneData.position) times.push(...Object.keys(boneData.position).map(parseFloat));
        if (boneData.rotation) times.push(...Object.keys(boneData.rotation).map(parseFloat));

        if (times.length === 0) continue;

        const boneMax = Math.max(...times);
        if (boneMax > maxTime) maxTime = boneMax;
    }

    return maxTime;
}

// Animation loop
const clock = new THREE.Clock();

function animateVRM() {
    if (!animationData || !vrm) return;

    const elapsedTime = clock.getElapsedTime();
    const duration = getAnimationDuration(animationData);

    const t = Math.min(elapsedTime, duration); // clamp to duration

    applyAnimation(vrm, animationData, nameMapData, t, duration);
    vrm.scene.updateMatrixWorld(true);
    renderer.render(scene, camera);

    if (elapsedTime >= duration) {
        // Uncomment if you want animation once, idle continiusly
        //animationData = idleAnimationData;
        clock.start();
    }

    requestAnimationFrame(animateVRM);
}


loader.load(
    '/static/_.vrm',
    (gltf) => {
        vrm = gltf.userData.vrm;


        // scale & position VRM to be visible
        vrm.scene.scale.set(0.5, 0.5, 0.5);
        vrm.scene.position.set(-1, 0, -1);
        scene.add(vrm.scene);

        requestAndPlayAnimation("Idle", true);

        console.log('VRM loaded:', vrm);
    },
    (progress) => console.log('Loading model...', 100.0 * (progress.loaded / progress.total), '%'),
    (error) => console.error(error)
);

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

export {requestAndPlayAnimation};