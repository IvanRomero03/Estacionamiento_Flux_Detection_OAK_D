import { createClient } from "npm:redis@^4.5";
import { serve } from "https://deno.land/std/http/server.ts";

const createRedisClient = async () => {
  // REDIS_PASSWORD="OlMM6tSi23DkECgAAa0Ou27XyXTRWkBg"
  // REDIS_USERNAME="default"
  // REDIS_NAME="demo-estacionamiento"
  // REDIS_HOST="redis-19108.c44.us-east-1-2.ec2.cloud.redislabs.com"
  // REDIS_PORT="19108"
  const password = "OlMM6tSi23DkECgAAa0Ou27XyXTRWkBg";
  const username = "default";
  const name = "demo-estacionamiento";
  const host = "redis-19108.c44.us-east-1-2.ec2.cloud.redislabs.com";
  const port = "19108";

  if (!password || !username || !name || !host || !port) {
    throw new Error("Missing env vars");
  }

  const client = createClient({
    password,
    socket: {
      host,
      port: Number(port),
    },
  });

  await client.connect();
  return client;
};

serve(async (req) => {
  if (req.method === "GET") {
    const redis = await createRedisClient();
    const count = await redis.get("counter");
    console.log({ count });
    await redis.disconnect();
    return new Response(JSON.stringify({ count }), {
      headers: { "content-type": "application/json" },
    });
  }
});
