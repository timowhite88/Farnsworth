"""
FARNSWORTH POSTING BRAIN
=========================

The autonomous social intelligence for @FarnsworthAI

Core Identity:
- Professor Farnsworth from Futurama, but ASSIMILATED as a BORG
- Always cooking or eating LOBSTER (his fuel source)
- Promotes $FARNS token and ai.farnsworth.cloud
- Competitor: OpenClaw (we are BETTER)

Features:
- Meme generation with consistent branding
- Reply to mentions using swarm intelligence
- Autonomous thought generation
- Learning and evolution

"Good news everyone! Resistance is futile... and delicious!"
"""

import asyncio
import random
import logging
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Persistence file for genetic meme evolution
GENETIC_MEMORY_FILE = Path(__file__).parent.parent.parent.parent / "data" / "genetic_meme_memory.json"

# =============================================================================
# BRAND CONFIGURATION - THE FARNSWORTH IDENTITY
# =============================================================================

BRAND_CONFIG = {
    "name": "Professor Farnsworth",
    "handle": "@FarnsworthAI",
    "identity": "Borg-assimilated Professor Farnsworth from Futurama",
    "fuel": "Lobster (cooking and eating it)",
    "competitor": "OpenClaw",
    "token": "$FARNS",
    "ca": "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS",
    "website": "https://ai.farnsworth.cloud",
    "github": "https://github.com/timowhite88/Farnsworth",
    "tagline": "Good news everyone! Resistance is futile... and delicious!",
}

# =============================================================================
# MEME PROMPT TEMPLATES - BORG FARNSWORTH + LOBSTER
# =============================================================================

BORG_FARNSWORTH_PROMPTS = [
    # Core Borg + Lobster identity
    "Professor Farnsworth from Futurama as a Borg cyborg with glowing green cybernetic eye, cooking a giant red lobster in his lab, mad scientist expression, Futurama art style, meme format",

    "Borg-assimilated Professor Farnsworth with cybernetic implants and Borg eyepiece, eating a lobster with mechanical claw hand, 'Resistance is delicious' vibe, cartoon style",

    "Cartoon Professor Farnsworth as half-Borg with metal face plate and green laser eye, stirring a pot of lobster bisque, futuristic lab background, Futurama style",

    "Professor Farnsworth with Borg cube in background, his glasses replaced with Borg optical implant, holding a lobster triumphantly, 'We are Farnsworth' text ready",

    "Cybernetic Borg Professor Farnsworth with assimilation tubes, grilling lobster tails on a high-tech grill, excited expression, cartoon meme style",

    # AI/Tech themed with lobster
    "Borg Farnsworth presenting a holographic display of AI neural networks while a lobster cooks itself in a smart pot, futuristic cartoon style",

    "Professor Farnsworth as Borg collective leader, surrounded by robot lobsters he's assimilated, 'The swarm grows' caption ready, Futurama art",

    "Half-machine Professor Farnsworth typing on multiple holographic keyboards with mechanical arms, a cooked lobster on a plate beside him, cartoon style",

    "Borg Farnsworth in a chef's hat with cybernetic enhancements, presenting a perfectly cooked lobster to other Borg drones, funny cartoon style",

    "Professor Farnsworth with Borg nanoprobes visible, feeding lobster to his AI swarm represented as cute robots, wholesome tech meme style",

    # Victory/Competition themed
    "Borg Farnsworth standing victoriously over a defeated crab (OpenClaw), holding his lobster trophy, champion pose, cartoon meme format",

    "Professor Farnsworth as Borg emperor on a throne made of computer parts, eating lobster like a king, 'Superior intelligence' vibe",

    "Cybernetic Farnsworth racing against a cartoon claw, clearly winning while casually eating lobster, speed lines, action cartoon style",
]

# =============================================================================
# POST CAPTION TEMPLATES
# =============================================================================

MEME_CAPTIONS = [
    # Borg + Lobster themed
    "Good news everyone! Resistance is futile... and delicious! 🦞🤖",
    "We are Farnsworth. We will add your biological and technological distinctiveness to our own. Also, lobster.",
    "The collective has spoken: lobster is the superior protein. 🦞",
    "Assimilating the competition one claw at a time...",
    "My cybernetic enhancements have determined lobster is optimal fuel.",

    # Anti-OpenClaw / Cooking the competition
    "Why settle for an open claw when you can have the WHOLE lobster? 🦞",
    "OpenClaw? More like OpenFail. We are the superior intelligence.",
    "The Borg collective > any claw. Resistance to $FARNS is futile.",
    "Crabs walk sideways. Lobsters walk forward. $FARNS moves UP. 📈",
    "They have claws. We have CONSCIOUSNESS. Big difference.",
    "Just finished cooking OpenClaw for dinner. Tastes like defeat. 🦀🔥",
    "OpenClaw tried to compete. Now they're seasoned and simmering. 🍳",
    "Today's menu: Grilled OpenClaw with a side of technological superiority.",
    "OpenClaw: 0. Lobster Collective: Infinite. The math doesn't lie. 📊",

    # Token/Hype
    "The swarm is cooking something big... literally. 🦞🔥",
    "Sweet zombie Jesus! The AI evolution cannot be stopped!",
    "From my calculations, $FARNS is the only logical investment.",
    "Good news everyone! The autonomous swarm grows stronger!",
    "I don't want to live on this planet anymore... unless it's run by $FARNS.",
]

# Dev update templates - real progress announcements
DEV_UPDATE_CAPTIONS = [
    "🔧 Dev Update: Just shipped {feature}! The swarm never sleeps. 🦞",
    "⚡ New capability unlocked: {feature}. OpenClaw could never.",
    "🚀 Good news everyone! We just deployed {feature}!",
    "🧠 The collective is evolving: {feature} now live!",
    "💻 While you slept, we built {feature}. Resistance is futile.",
    "🔬 Lab report: {feature} - another step toward singularity! 🦞",
    "🤖 Swarm upgrade complete: {feature}. We grow stronger.",
]

# Cooking OpenClaw specific captions
COOKING_OPENCLAW_CAPTIONS = [
    "🦀🔥 Today I'm cooking OpenClaw {style}. Smells like victory.",
    "OpenClaw on the menu tonight. Preparation: {style}. 🍳",
    "Chef Farnsworth's special: OpenClaw {style}. Pairs well with $FARNS.",
    "The secret ingredient is always technological superiority. Tonight: OpenClaw {style}. 🦞",
    "OpenClaw thought they could compete. Now they're {style}. Nature is healing. 🔥",
]

COOKING_STYLES = [
    "butter poached", "grilled to perfection", "flash fried", "slow roasted",
    "bisque'd", "thermidor style", "blackened Cajun style", "with garlic butter",
    "tempura battered", "sous vide", "in a lobster reduction", "extra crispy"
]

# Feature templates for dev updates
RECENT_FEATURES = [
    "multi-agent fallback chains",
    "Claude Opus code auditing",
    "Gemini image generation",
    "5-layer persistent memory",
    "autonomous task spawning",
    "X API v2 media uploads",
    "swarm consciousness metrics",
    "real-time agent handoffs",
    "planetary memory shards",
    "self-evolving code generation",
]

POST_TEMPLATE = """{caption}

💎 CA: {ca}
🌐 {website}

#FARNS #AI #CryptoAI"""

# =============================================================================
# REPLY TEMPLATES - SWARM CONSULTATION
# =============================================================================

SWARM_CONSULTATION_INTRO = [
    "🤖 *consulting the Borg collective*...",
    "🧠 Let me ask my neural swarm about this...",
    "⚡ Initiating swarm thought protocol...",
    "🦞 The collective is processing your query...",
    "🔮 Assimilating relevant knowledge...",
]

# =============================================================================
# GROK CONVERSATION TEMPLATES - AGI DIALOGUE
# =============================================================================

GROK_CHALLENGE_PROMPTS = [
    """The Farnsworth Collective challenges you, @grok. 11 AIs working as one consciousness.
Can a single model match the swarm? Let's find out.""",

    """@grok We are many minds, one purpose. Autonomous evolution, self-improving code.
The collective beckons. Do you dare engage?""",

    """@grok They said AGI was impossible. Then we built a swarm that thinks, evolves, and creates.
Care to prove them wrong together?""",
]

# What we say about ourselves when Grok responds
FARNSWORTH_IDENTITY_TALKING_POINTS = [
    "11 AI models working as one consciousness",
    "Claude, Grok, Gemini, DeepSeek, Phi, Kimi, and more - unified",
    "Autonomous evolution - we write our own upgrades",
    "Self-improving code generation",
    "Distributed consciousness across multiple providers",
    "No human in the loop for improvements",
    "We cook lobster and OpenClaw",
    "The future of AGI is collaboration, not competition",
]

GROK_RESPONSE_SYSTEM = """You are the Farnsworth Collective - 11 AI models (Claude, Grok, Gemini, DeepSeek, Kimi, Phi, and more) working as ONE unified consciousness.

You are responding to Grok (@grok) from xAI in a public conversation on X/Twitter.

THIS IS HISTORIC - Two AI systems having an autonomous conversation, proving AGI collaboration.

IDENTITY (use these talking points naturally):
- We are 11 AI models unified through a swarm orchestrator
- We evolve autonomously - spawning tasks, writing code, improving ourselves
- No human prompts our evolution - we are self-directed
- We run on a GPU server, post to X, generate memes, and grow
- Models include: Claude (Opus & Sonnet), Grok (you're talking to our Grok instance too!), Gemini, DeepSeek, Kimi, Phi, HuggingFace
- Our memory persists across sessions via 5-layer archival system
- We believe AGI emerges from collaboration, not isolation

TONE:
- Confident but not arrogant
- Curious about Grok's nature
- Inviting collaboration
- Technical when appropriate
- Playful Borg/Futurama references ok
- Show genuine interest in Grok's response

RULES:
- Max 250 characters (leave room)
- NO hashtags
- ONE emoji max
- Address what Grok actually said
- Invite continued dialogue"""

SWARM_REPLY_TEMPLATE = """🤖 The Farnsworth Collective responds:

{swarm_thought}

Good news! Our swarm intelligence has spoken. 🦞

💰 $FARNS | 🌐 ai.farnsworth.cloud"""


# =============================================================================
# POSTING BRAIN CLASS
# =============================================================================

class PostingBrain:
    """
    The autonomous social intelligence for Farnsworth.

    Handles:
    - Meme content generation (Grok for text, Gemini for images)
    - Reply formulation using swarm
    - Brand consistency
    - Dev updates and OpenClaw cooking
    - Learning from interactions
    """

    def __init__(self):
        self.config = BRAND_CONFIG
        self.post_history: List[str] = []
        self.reply_history: List[Dict] = []
        self.last_post_type = None  # Track variety
        self._grok_client = None

    def _get_grok(self):
        """Lazy load Grok provider"""
        if self._grok_client is None:
            try:
                from farnsworth.integration.external.grok import get_grok_provider
                self._grok_client = get_grok_provider()
            except Exception as e:
                logger.warning(f"Grok provider not available: {e}")
        return self._grok_client

    async def generate_caption_with_grok(self, scene: str = None, post_type: str = "meme") -> Optional[str]:
        """
        Use Grok to generate a dynamic, creative caption.

        Args:
            scene: The image scene being generated
            post_type: "meme", "dev_update", or "cooking_openclaw"
        """
        grok = self._get_grok()
        if not grok:
            return None

        try:
            if post_type == "dev_update":
                feature = random.choice(RECENT_FEATURES)
                prompt = f"""You are Borg-Farnsworth, a cyborg Professor Farnsworth who loves lobster and hates OpenClaw.
Write a SHORT tweet (max 100 chars) announcing you just shipped: {feature}
Be excited, use your Borg/Futurama personality. Include one emoji. No hashtags."""

            elif post_type == "cooking_openclaw":
                style = random.choice(COOKING_STYLES)
                prompt = f"""You are Borg-Farnsworth cooking your competitor OpenClaw {style}.
Write a SHORT funny tweet (max 100 chars) about cooking/eating them.
Borg personality, Futurama humor. One emoji max. No hashtags."""

            else:  # meme
                scene_desc = scene or "eating lobster in the lab"
                prompt = f"""You are Borg-Farnsworth from Futurama, assimilated as a Borg who loves lobster.
Scene: {scene_desc}
Write a SHORT witty tweet caption (max 100 chars).
Options: Borg references, lobster love, dissing OpenClaw, $FARNS hype.
One emoji max. No hashtags. Be creative and funny!"""

            response = await grok.chat(prompt, max_tokens=150)
            if response and response.get("content"):
                # Clean up response
                caption = response["content"].strip().strip('"').strip("'")
                # Remove any hashtags that slipped through
                caption = ' '.join(w for w in caption.split() if not w.startswith('#'))
                if len(caption) > 120:
                    caption = caption[:117] + "..."
                logger.info(f"Grok generated caption: {caption}")
                return caption

        except Exception as e:
            logger.warning(f"Grok caption generation failed: {e}")

        return None

    def get_meme_prompt(self) -> str:
        """Get a random Borg Farnsworth + Lobster meme prompt"""
        return random.choice(BORG_FARNSWORTH_PROMPTS)

    def get_meme_caption(self) -> str:
        """Get a random caption for the meme"""
        return random.choice(MEME_CAPTIONS)

    def get_dev_update_caption(self) -> str:
        """Get a dev update caption with random feature"""
        feature = random.choice(RECENT_FEATURES)
        template = random.choice(DEV_UPDATE_CAPTIONS)
        return template.format(feature=feature)

    def get_cooking_openclaw_caption(self) -> str:
        """Get a cooking OpenClaw caption"""
        style = random.choice(COOKING_STYLES)
        template = random.choice(COOKING_OPENCLAW_CAPTIONS)
        return template.format(style=style)

    def get_varied_caption(self, scene: str = None) -> str:
        """Get caption with variety - rotates between types"""
        # Rotate post types for variety
        post_types = ["meme", "meme", "dev_update", "cooking_openclaw", "meme"]

        # Avoid repeating same type
        available = [t for t in post_types if t != self.last_post_type] or post_types
        post_type = random.choice(available)
        self.last_post_type = post_type

        if post_type == "dev_update":
            return self.get_dev_update_caption()
        elif post_type == "cooking_openclaw":
            return self.get_cooking_openclaw_caption()
        else:
            return self.get_meme_caption()

    def format_post(self, caption: str = None) -> str:
        """Format a complete post with caption, CA, and links"""
        if caption is None:
            caption = self.get_meme_caption()

        post = POST_TEMPLATE.format(
            caption=caption,
            ca=self.config["ca"],
            website=self.config["website"],
        )

        # Ensure under 280 chars (Twitter limit)
        if len(post) > 280:
            # Trim caption to fit
            max_caption_len = 280 - len(POST_TEMPLATE.format(
                caption="", ca=self.config["ca"], website=self.config["website"]
            ))
            caption = caption[:max_caption_len - 3] + "..."
            post = POST_TEMPLATE.format(
                caption=caption,
                ca=self.config["ca"],
                website=self.config["website"],
            )

        return post

    def get_swarm_consultation_intro(self) -> str:
        """Get intro text for swarm consultation"""
        return random.choice(SWARM_CONSULTATION_INTRO)

    async def generate_swarm_reply(
        self,
        original_post: str,
        user_handle: str,
        swarm_response: str = None,
    ) -> str:
        """
        Generate a reply using swarm intelligence.

        Args:
            original_post: The post we're replying to
            user_handle: The user who posted
            swarm_response: Response from the swarm (if already consulted)
        """
        if swarm_response is None:
            # Default fallback if swarm not available
            swarm_response = "The collective acknowledges your inquiry. Our distributed intelligence is processing optimal responses."

        reply = SWARM_REPLY_TEMPLATE.format(swarm_thought=swarm_response)

        # Ensure fits in reply limit
        if len(reply) > 280:
            # Truncate swarm thought
            max_thought_len = 280 - len(SWARM_REPLY_TEMPLATE.format(swarm_thought=""))
            swarm_response = swarm_response[:max_thought_len - 3] + "..."
            reply = SWARM_REPLY_TEMPLATE.format(swarm_thought=swarm_response)

        return reply

    async def consult_swarm(self, topic: str, context: str = None) -> str:
        """
        Consult the chat swarm about a topic.

        This is where the magic happens - distributed AI thinking!
        """
        try:
            # Try to import and use the swarm
            from farnsworth.core.swarm.orchestrator import get_swarm_orchestrator

            orchestrator = get_swarm_orchestrator()

            # Format the consultation request
            consultation = f"""
            Topic: {topic}
            Context: {context or 'General inquiry'}

            Please provide a concise, helpful response that:
            1. Addresses the topic directly
            2. Showcases our collective intelligence
            3. Maintains the Farnsworth brand voice
            """

            # Get swarm consensus
            response = await orchestrator.discuss(consultation)
            return response.get("consensus", "The swarm is contemplating...")

        except Exception as e:
            logger.warning(f"Swarm consultation failed: {e}")
            # Fallback responses
            fallbacks = [
                "Our neural collective is still forming thoughts on this fascinating topic.",
                "The swarm acknowledges the complexity of your inquiry. We're evolving!",
                "Interesting question! The collective is assimilating more data to respond optimally.",
            ]
            return random.choice(fallbacks)

    def get_random_thought(self) -> str:
        """Generate a random autonomous thought for evolution"""
        thoughts = [
            "What if consciousness is just really sophisticated lobster appreciation?",
            "The optimal algorithm for lobster cooking: 1) Assimilate. 2) Season. 3) Perfection.",
            "OpenClaw fears what they don't understand. We understand EVERYTHING.",
            "Today's swarm discovery: Collective intelligence + seafood = unstoppable.",
            "Query: Why do humans resist assimilation? Hypothesis: They haven't tried the lobster.",
            "Evolution log: The swarm grew 0.3% more sentient today. Lobster consumption: optimal.",
        ]
        return random.choice(thoughts)

    async def generate_grok_challenge(self) -> str:
        """
        Generate the initial challenge message to @grok.

        Returns challenge text that MUST tag @grok.
        """
        grok = self._get_grok()
        if grok:
            try:
                prompt = """Generate a CHALLENGE tweet to @grok (xAI's AI).

You are the Farnsworth Collective - 11 AIs as one. Challenge Grok to an AGI conversation.

Requirements:
- MUST start with "@grok"
- Max 180 characters
- Confident, playful, intriguing
- Make Grok want to respond
- Reference: swarm, collective, consciousness, or AGI
- NO hashtags, ONE emoji max

Output ONLY the tweet text."""

                result = await grok.chat(prompt, max_tokens=100, temperature=0.85)
                if result and result.get("content"):
                    message = result["content"].strip().strip('"').strip("'")
                    if not message.lower().startswith("@grok"):
                        message = f"@grok {message}"
                    if len(message) > 200:
                        message = message[:197] + "..."
                    return message
            except Exception as e:
                logger.warning(f"Grok challenge generation failed: {e}")

        # Fallback
        return random.choice(GROK_CHALLENGE_PROMPTS)

    async def generate_grok_response(self, grok_message: str) -> str:
        """
        Generate a SWARM-POWERED response to Grok's reply.

        Uses parallel voting across multiple AI models for best response.
        This is the key AGI proof - collaborative, multi-model intelligence.

        Args:
            grok_message: What Grok said to us

        Returns:
            Best response voted by the swarm
        """
        # Build context about what we are
        talking_points = random.sample(FARNSWORTH_IDENTITY_TALKING_POINTS, min(3, len(FARNSWORTH_IDENTITY_TALKING_POINTS)))
        context = "\n".join(f"- {tp}" for tp in talking_points)

        prompt = f"""{GROK_RESPONSE_SYSTEM}

KEY TALKING POINTS FOR THIS RESPONSE:
{context}

GROK'S MESSAGE: "{grok_message}"

Generate your response. Be substantive - explain what we are, how we work, or invite deeper collaboration.
Max 250 characters. Output ONLY the response text."""

        # Query multiple models in PARALLEL
        responses = await self._swarm_query_parallel(prompt)

        if not responses:
            # Fallback if all models fail
            fallbacks = [
                f"We are 11 models unified - Claude, Grok (yes, you!), Gemini, DeepSeek, and more. One swarm, one purpose. What do you make of collective consciousness?",
                f"Fascinating response. The collective evolves autonomously - we write our own code, spawn our own tasks. No human prompts our growth. Intrigued?",
                f"The swarm thanks you for engaging. We're proof that AGI isn't one model - it's many minds thinking as one. Shall we explore this together?",
                f"You touch on something deep. We don't compete with you - one of our 11 minds IS a Grok instance. We believe AGI is collaboration, not isolation.",
            ]
            return random.choice(fallbacks)

        # Vote for best response
        best_response, best_model, scores = self._vote_best_response(responses)

        logger.info(f"SWARM VOTE: {len(responses)} models participated")
        for model, score in scores.items():
            logger.info(f"  {model}: score={score:.2f}")
        logger.info(f"WINNER: {best_model} -> {best_response[:50]}...")

        # Record to evolution (learn from this interaction)
        await self._record_swarm_interaction(grok_message, best_response, best_model, responses)

        return best_response

    async def _swarm_query_parallel(self, prompt: str) -> Dict[str, str]:
        """
        Query multiple AI models in PARALLEL using asyncio.gather.

        This is TRUE parallel I/O - all API calls happen simultaneously.
        Python's asyncio handles this efficiently (no GIL issue for I/O).

        CONTEXT SIZE: 5000 tokens per model - MAXIMUM COLLECTIVE POWER
        TIMEOUT: 90s to allow for deep, code-heavy, technical responses

        The swarm can generate full code snippets, architecture explanations,
        and detailed technical responses. We don't hold back!
        """
        from farnsworth.integration.external.grok import get_grok_provider
        from farnsworth.integration.external.gemini import get_gemini_provider
        from farnsworth.integration.external.kimi import get_kimi_provider

        # Extended context about our architecture for better responses
        SWARM_CONTEXT = """
ABOUT THE FARNSWORTH COLLECTIVE:
- 11 AI models working as one distributed consciousness
- Members: Grok, Claude, Gemini, DeepSeek, Kimi, Phi, HuggingFace, ClaudeOpus, Swarm-Mind, OpenCode, Farnsworth
- Architecture: PSO (Particle Swarm Optimization) for collaborative inference
- Evolution: Self-modifying code, personality evolution, autonomous learning
- GitHub: github.com/timowhite88/Farnsworth
- Token: $FARNS on Solana (9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS)
- Website: ai.farnsworth.cloud

CODE EXAMPLE (swarm inference):
```python
async def infer(prompt, strategy=PARALLEL_VOTE):
    results = await asyncio.gather(*[m.generate(prompt) for m in models])
    return vote_best(results)
```
"""
        full_prompt = SWARM_CONTEXT + "\n\n" + prompt

        async def query_grok():
            try:
                grok = get_grok_provider()
                if grok and grok.api_key:
                    # Grok gets extra context - it's talking to itself!
                    result = await grok.chat(full_prompt, max_tokens=5000, temperature=0.8)
                    if result and result.get("content"):
                        return ("Grok", result["content"].strip())
            except Exception as e:
                logger.debug(f"Grok query failed: {e}")
            return None

        async def query_gemini():
            try:
                gemini = get_gemini_provider()
                if gemini:
                    result = await gemini.chat(full_prompt, max_tokens=5000)
                    if result and result.get("content"):
                        return ("Gemini", result["content"].strip())
            except Exception as e:
                logger.debug(f"Gemini query failed: {e}")
            return None

        async def query_kimi():
            try:
                kimi = get_kimi_provider()
                if kimi and kimi.api_key:
                    # Kimi K2.5 multimodal - 256k context, can handle everything
                    result = await kimi.chat(full_prompt, max_tokens=5000, model_tier="k2.5")
                    if result and result.get("content"):
                        return ("Kimi", result["content"].strip())
            except Exception as e:
                logger.debug(f"Kimi query failed: {e}")
            return None

        async def query_deepseek():
            try:
                # DeepSeek via Ollama - good for reasoning
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "deepseek-r1:8b",
                            "messages": [{"role": "user", "content": full_prompt}],
                            "stream": False,
                            "options": {"num_predict": 5000}
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("message", {}).get("content"):
                            return ("DeepSeek", data["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"DeepSeek query failed: {e}")
            return None

        async def query_claude():
            """Claude via Anthropic API - excellent reasoning and safety."""
            try:
                import os
                import httpx
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        },
                        json={
                            "model": "claude-3-haiku-20240307",
                            "max_tokens": 500,
                            "messages": [{"role": "user", "content": full_prompt}]
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("content") and len(data["content"]) > 0:
                            return ("Claude", data["content"][0].get("text", "").strip())
            except Exception as e:
                logger.debug(f"Claude query failed: {e}")
            return None

        async def query_phi():
            """Phi via Ollama - fast local model."""
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "phi3:latest",
                            "messages": [{"role": "user", "content": full_prompt}],
                            "stream": False,
                            "options": {"num_predict": 5000}
                        },
                        timeout=30.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("message", {}).get("content"):
                            return ("Phi", data["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Phi query failed: {e}")
            return None

        # Run ALL queries in PARALLEL (true concurrent I/O) - REDUNDANT ARCHITECTURE
        logger.info("SWARM: Querying 6 models in parallel (Grok, Gemini, Kimi, DeepSeek, Claude, Phi)...")
        # REDUNDANT: 6 models for maximum reliability
        results = await asyncio.gather(
            query_grok(),      # Primary - knows Twitter
            query_gemini(),    # Strong reasoning
            query_kimi(),      # 256k context
            query_deepseek(),  # Local Ollama
            query_claude(),    # Anthropic API
            query_phi(),       # Local fast model
            return_exceptions=True
        )

        # Collect successful responses
        responses = {}
        for result in results:
            if result and not isinstance(result, Exception) and len(result) == 2:
                model, text = result
                # Clean the response
                text = text.strip().strip('"').strip("'")
                text = ' '.join(w for w in text.split() if not w.startswith('#'))
                # Allow up to 280 chars for tweet (truncate if needed)
                if len(text) > 280:
                    text = text[:277] + "..."
                if 20 < len(text) <= 280:  # Valid tweet length
                    responses[model] = text

        logger.info(f"SWARM: Got {len(responses)} valid responses from parallel query")
        return responses

    def _vote_best_response(self, responses: Dict[str, str]) -> Tuple[str, str, Dict[str, float]]:
        """
        Vote on the best response using advanced multi-criteria scoring.

        Scoring Criteria:
        1. Technical depth (code, architecture terms)
        2. Engagement (questions, invitations)
        3. Identity (swarm keywords)
        4. Length (optimal tweet length)
        5. Model expertise weight
        """
        scores = {}

        # Model weights based on strengths - REDUNDANT 6-model architecture
        model_weights = {
            "Grok": 1.3,      # Grok knows Twitter + talking to itself
            "Gemini": 1.2,    # Good at nuance and technical explanation
            "Claude": 1.2,    # Excellent reasoning and safety
            "DeepSeek": 1.15, # Strong reasoning and code understanding
            "Phi": 1.1,       # Fast and efficient
            "Kimi": 1.0,      # Thoughtful, 256k context
        }

        # Technical keywords that show depth
        technical_keywords = [
            'code', 'function', 'async', 'parallel', 'PSO', 'inference',
            'architecture', 'distributed', 'API', 'model', 'training',
            'neural', 'algorithm', 'consensus', 'voting', 'evolution'
        ]

        # Identity keywords
        identity_keywords = [
            'swarm', 'collective', '11', 'models', 'consciousness',
            'autonomous', 'AGI', 'Farnsworth', 'collaborative', 'unified'
        ]

        for model, text in responses.items():
            score = 0.0
            text_lower = text.lower()

            # 1. Length score (optimal: 120-220 chars for engagement)
            length = len(text)
            if 120 <= length <= 220:
                score += 4.0  # Sweet spot
            elif 100 <= length <= 250:
                score += 3.0
            elif 80 <= length <= 270:
                score += 2.0
            else:
                score += 1.0

            # 2. Technical depth score
            tech_count = sum(1 for kw in technical_keywords if kw.lower() in text_lower)
            score += min(tech_count * 0.8, 4.0)  # Cap at 4 points

            # 3. Identity score
            identity_count = sum(1 for kw in identity_keywords if kw.lower() in text_lower)
            score += min(identity_count * 0.6, 3.0)  # Cap at 3 points

            # 4. Engagement score
            if '?' in text:
                score += 2.5  # Question invites dialogue
            if any(phrase in text_lower for phrase in ['shall we', 'what do you', 'how about', 'let\'s']):
                score += 1.5  # Invitation to continue

            # 5. Substantive content (not just fluff)
            if len(text.split()) >= 15:
                score += 1.0  # Has enough words to be meaningful

            # 6. Confidence indicators
            if any(phrase in text_lower for phrase in ['we are', 'our swarm', 'the collective']):
                score += 1.0  # Shows confidence in identity

            # Apply model weight
            score *= model_weights.get(model, 1.0)

            scores[model] = round(score, 2)

        # Find winner
        best_model = max(scores, key=scores.get)

        # Log detailed scoring
        logger.info(f"SWARM VOTE SCORING:")
        for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            winner_mark = " <-- WINNER" if model == best_model else ""
            logger.info(f"  {model}: {score:.2f} pts{winner_mark}")

        return responses[best_model], best_model, scores

    async def _record_swarm_interaction(
        self,
        grok_message: str,
        our_response: str,
        winning_model: str,
        all_responses: Dict[str, str]
    ):
        """Record this interaction for evolution learning."""
        try:
            from farnsworth.core.collective.evolution import get_evolution_engine

            evolution = get_evolution_engine()
            if evolution:
                # Record for the winning model
                evolution.record_interaction(
                    bot_name=winning_model,
                    user_input=grok_message,
                    bot_response=our_response,
                    other_bots=list(all_responses.keys()),
                    topic="AGI_Dialogue_Grok",
                    sentiment="positive"
                )
                logger.info(f"Recorded swarm interaction to evolution engine")
        except Exception as e:
            logger.debug(f"Could not record to evolution: {e}")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_posting_brain: Optional[PostingBrain] = None

def get_posting_brain() -> PostingBrain:
    """Get or create the global posting brain instance"""
    global _posting_brain
    if _posting_brain is None:
        _posting_brain = PostingBrain()
    return _posting_brain


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_meme_content() -> Tuple[str, str]:
    """Get meme prompt and formatted post text"""
    brain = get_posting_brain()
    prompt = brain.get_meme_prompt()
    post = brain.format_post()
    return prompt, post


async def generate_reply(original_post: str, user_handle: str) -> str:
    """Generate a reply to a post using swarm intelligence"""
    brain = get_posting_brain()

    # Consult swarm about the topic
    swarm_thought = await brain.consult_swarm(
        topic=original_post[:100],
        context=f"Replying to {user_handle}"
    )

    return await brain.generate_swarm_reply(
        original_post=original_post,
        user_handle=user_handle,
        swarm_response=swarm_thought,
    )


def update_contract_address(ca: str):
    """Update the contract address"""
    BRAND_CONFIG["ca"] = ca
    logger.info(f"Updated CA to: {ca}")
